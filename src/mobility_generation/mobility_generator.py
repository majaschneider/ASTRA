from pathlib import Path
import copy
from typing import Iterator
from tqdm import tqdm
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ray
from skmob import TrajDataFrame

from src.database.database import Database
from src.config import (
    TessellationConfig,
    RunConfig,
    GeneralConfig,
)
from src.survey_processing.loading.survey_loading import SurveyLoader
from src.survey_processing.transforms.dataset_adapter import DatasetAdapter, load_or_create_dataset
from src.survey_processing.sequence_model.sequence_model import SequenceModel
from src.survey_processing.loading.travel_survey_encoding import SurveyEncoding
from src.survey_processing.transforms.scaler import Scaler
from src.tessellation.data_classes.demographic import DemographicsData, Demographic
from src.tessellation.tessellation.tessellation import Tessellation
from src.tessellation.loading.osm_data import OSMData
from src.mobility_generation.data_classes.agent import Agent
from src.mobility_generation.data_classes.route import Route
from src.mobility_generation.encoding.embedding_collection import EmbeddingCollection
from src.mobility_generation.poi_route.poi_selector import PoiSelector
from src.tessellation.data_classes.pois import Poi
from src.mobility_generation.cell_route.semantic_depr import SemanticDEPR
from src.mobility_generation.utils.df_creation import to_df
from src.helper.helper import CellEmptyException


def set_cell_route_for_agent(
    epr: SemanticDEPR,
    agent: Agent,
    start_cell_idx: int,
    tessellation: Tessellation,
) -> Agent:
    """Calculate and set the agent's route."""
    agent.start_cell_idx = start_cell_idx
    print('###### calculate agent for start cell idx', start_cell_idx)
    print(agent.agenda.persona_features)
    print('activities', agent.agenda.activities)
    print('nr segments', agent.agenda.number_of_segments)
    print('travel', agent.agenda.travel)
    print('current segment', agent.current_segment)

    # Create first element in the agent's route corresponding to the start cell of the agent
    agent.append_route_element(epr.calculate_next_route_element(agent=agent, tessellation=tessellation))

    while True:
        try:
            agent.next_segment()
            print('current segment', agent.current_segment)
            agent.append_route_element(epr.calculate_next_route_element(agent=agent, tessellation=tessellation))
        except StopIteration:
            break

    return agent


def calculate_poi_route_from_cell_route(route: Route, tessellation: Tessellation, poi_selector: PoiSelector) -> list[Poi]:
    # Prepare dictionary for storing POIs
    pois = {}

    # Select POI for each segment that was not a return
    for segment_id in route.segment_ids_without_returns():
        cell_idx = route.cell_idxs[segment_id]
        prev_cell_idx = route.cell_idxs[segment_id - 1] if segment_id > 0 else None

        osm_pois = tessellation.get_cell_poi_data(cell_idx=cell_idx)
        embedding_id = route.embedding_ids[segment_id]
        pois[segment_id] = poi_selector.select_osm_poi(
            previous_cell_idx=prev_cell_idx,
            embedding_id=embedding_id,
            poi_candidates=osm_pois,
            tessellation=tessellation,
        )

    # Handle return segments
    for segment_id, return_id in route.returns.items():
        pois[segment_id] = pois[return_id]

    # Return the POIs sorted by segment ID
    return [pois[segment_id] for segment_id in sorted(pois)]


class MobilityGenerator:
    """Generator to create synthetic routes from dataset or sequence model."""
    config: GeneralConfig
    run_config: RunConfig
    tessellation_config: TessellationConfig

    survey_encoding: SurveyEncoding
    loading_modules: tuple[SurveyLoader, ...] = None
    survey_paths: tuple[Path, ...] = None

    sequence_model: SequenceModel

    tessellation: Tessellation
    demographic_data: DemographicsData
    embedding_collection: EmbeddingCollection
    osm_data: OSMData

    cell_selector: SemanticDEPR
    poi_selector: PoiSelector

    rng: np.random.Generator

    database: Database

    def __init__(
        self,
        config: GeneralConfig,
        run_config: RunConfig,
        database: Database,
        survey_encoding: SurveyEncoding,
        loading_modules: tuple[SurveyLoader, ...] = None,
        survey_paths: tuple[Path, ...] = None,
        num_gpus: int = 1,
    ) -> None:
        self.config = config
        self.run_config = run_config
        self.database = database
        self.survey_encoding = survey_encoding
        self.loading_modules = loading_modules
        self.survey_paths = survey_paths
        self.rng = np.random.default_rng(seed=config.seed)

        # initialize environment for parallel processing with ray
        if not ray.is_initialized():
            ray.init()

        # initialize data structure for retrieving area demographics
        self.demographic_data = DemographicsData(
            data_paths=config.paths_demographic_data,
            categorical_persona_feature_names=config.survey_processing_config.categorical_persona_feature_names,
            real_persona_feature_names=config.survey_processing_config.real_persona_feature_names,
        )

        # load sentence transformer and load index
        # functions as one actor when processing in parallel
        self.embedding_collection = EmbeddingCollection.remote(
            survey_encoding=self.survey_encoding,
            rng=self.rng,
            index_path=config.path_embedding_index,
            use_weighted_average_search=config.route_creation_config.use_weighted_average_search,
            weight_impact_on_average=config.route_creation_config.weight_impact_on_average,
        )

        self.tessellation = Tessellation(
            area=self.run_config.area,
            embedding_collection=self.embedding_collection,
            demographic_data=self.demographic_data,
            tessellation_config=self.config.tessellation_config,
            run_config=self.run_config,
            config=self.config,
            database=self.database
        )

        # initialize selector class for osm pois
        self.poi_selector = PoiSelector(rng=self.rng)

        # initialize selector class for cells
        self.cell_selector = SemanticDEPR.from_config(
            survey_encoding=survey_encoding,
            route_creation_config=config.route_creation_config,
            rng=self.rng
        )

    def _create_agents_from_sequence_model(
            self,
            demographic: Demographic,
            starting_timestamp: datetime,
            starting_timestamp_std: timedelta,
    ) -> Iterator[Agent]:
        """Create agents with given demographic from sequence model."""
        if self.sequence_model is None:
            self.load_or_create_sequence_model()

        agendas = self.sequence_model(
            demographic=demographic,
            starting_timestamp=starting_timestamp,
            starting_timestamp_std=starting_timestamp_std,
        )

        agent_class = self.cell_selector.get_agent_type()
        for agenda in agendas:
            yield agent_class.from_agenda(agenda=agenda)

    def load_or_create_sequence_model(self):
        # load or create scaler
        scaler = Scaler(
            survey_processing_config=self.config.survey_processing_config,
            reload=self.config.retrain_sequence_model and self.config.force_reload_dataset,
            survey_encoding=self.survey_encoding,
        )

        # create factory for creating sequence datasets
        sequence_dataset_factory = DatasetAdapter.from_config(
            scaler=scaler,
            survey_encoding=self.survey_encoding,
            survey_processing_config=self.config.survey_processing_config,
            seed=self.config.seed,
            rng=self.rng,
        )

        # if sequence model is retrained, load or create survey dataset
        if self.config.retrain_sequence_model:
            training_datasets = load_or_create_dataset(
                path_processed_dataset=self.config.path_processed_dataset,
                force_reload_dataset=self.config.force_reload_dataset,
                survey_paths=self.survey_paths,
                loading_modules=self.loading_modules,
                dataset_factory=sequence_dataset_factory,
            )
        else:
            training_datasets = None

        self.sequence_model = SequenceModel(
            model_config=self.config.model_config,
            training_config=self.config.training_config,
            training_datasets=training_datasets,
            retrain_sequence_model=self.config.retrain_sequence_model,
            dataset_adapter=sequence_dataset_factory,
        )
        del training_datasets
        print('SequenceModel loaded.')

    def load_agents_dataset(self, nr_agents: int = -1) -> Iterator[Agent]:
        print('Load agendas from dataset.')
        if nr_agents > 0:
            print('Limit dataset to', nr_agents, 'agents.')

        mtus_loader = self.loading_modules[0]
        agenda_frequency_iterator = mtus_loader.load_from_paths(
            path=self.survey_paths[0],
            nr_agents=nr_agents
        )
        agent_class = self.cell_selector.get_agent_type()
        for agenda, frequency in agenda_frequency_iterator:
            for i in range(frequency):
                agent = agent_class.from_agenda(agenda=agenda)
                yield agent

    def generate(self, run_config: RunConfig, config_id: int, nr_surveys: int = -1) -> Iterator[TrajDataFrame]:
        """
        Generate a route for each agent. If nr_surveys is specified, the requested number of surveys is loaded from the
        travel survey data set.
        """
        demographics_iterator = self.demographic_data.create_demographics(
            areas=self.tessellation.cells_with_pois,
            nr_agents=run_config.num_agents,
        )

        # load nr_surveys from dataset (or all surveys if nr_surveys < 0)
        dataset = list(self.load_agents_dataset(nr_agents=nr_surveys))
        print("Number of agents loaded from MTUS data set:", len(dataset))
        # index the agents and their demographics
        mtus_persona_features = pd.DataFrame(data=[agent.agenda.persona_features for agent in dataset], columns=['SEX', 'AGE'])

        for cell_idx, demographic in tqdm(
            enumerate(demographics_iterator),
            total=len(self.tessellation.cells_with_pois),
            unit="cell",
            desc="Calculating agents and rough cell routes for each cell",
        ):
            if demographic.nr_simulated_agents == 0:
                continue

            # randomly draw agents with the required demographics from the dataset according to their relative frequency
            rng = np.random.default_rng(seed=self.config.seed)
            agent_idxs = rng.choice(
                a=len(demographic.persona_feature_distribution),
                size=demographic.nr_simulated_agents,
                p=demographic.persona_feature_distribution.relative_freq
            )

            # try to find persona features in dataset
            agents = []
            for agent_idx in agent_idxs:
                target_persona_features = demographic.persona_feature_distribution.iloc[agent_idx]
                agent_candidates = mtus_persona_features[mtus_persona_features.SEX == int(target_persona_features.SEX)].copy()
                agent_candidates['age_diff'] = np.abs(agent_candidates.AGE - int(target_persona_features.AGE))
                # select random person from the data set with matching age if available
                agents_with_exact_age_match = agent_candidates[agent_candidates['age_diff'] == 0]
                if len(agents_with_exact_age_match) > 0:
                    # pick with uniform distribution
                    random_idx = rng.choice(a=range(len(agents_with_exact_age_match)), size=1, p=None)[0]
                    agent_idx = agents_with_exact_age_match.iloc[random_idx].name
                else:
                    agents_with_approximate_age_match = agent_candidates.sort_values(by='age_diff', ascending=True)
                    # select the agent from the data set with the lowest age difference
                    if len(agents_with_approximate_age_match) > 0:
                        agent_idx = agents_with_approximate_age_match.iloc[0].name
                    else:
                        raise Exception('Not enough agents loaded to retrieve required demographics')
                agent = copy.deepcopy(dataset[agent_idx])
                agents.append(agent)
            agents_iterator = (agent for agent in agents)

            # calculate rough routes (on cell level) for all agents that start their route in this cell
            # if there are no POIs in a cell but there are agents to be simulated, throw a warning and ignore this cell
            try:
                agents_with_route = []
                for agent in agents_iterator:
                    try:
                        agent_with_route = set_cell_route_for_agent(
                            epr=self.cell_selector,
                            agent=agent,
                            start_cell_idx=cell_idx,
                            tessellation=self.tessellation,
                        )
                        agents_with_route.append(agent_with_route)
                    except Exception as e:
                        print(e)
                        print('Agent will be skipped:')
                        print(agent)
                        continue
            except CellEmptyException as e:
                print(e)
                print('Cell idx', cell_idx, 'with cell_id ', self.tessellation.cell_ids_with_pois[cell_idx],
                      'will be skipped.', demographic.nr_simulated_agents, 'agents will not be created).')
                continue

            for i, agent in tqdm(
                enumerate(agents_with_route),
                total=len(agents_with_route),
                unit="agent",
                desc=f"Calculating detailed POI routes for each agent in cell idx {cell_idx} "
                     f"(cell_id {self.tessellation.cell_ids_with_pois[cell_idx]})",
            ):
                poi_route = calculate_poi_route_from_cell_route(
                    route=agent.history,
                    tessellation=self.tessellation,
                    poi_selector=self.poi_selector,
                )
                # add agent to database
                agent_id = self.database.append_astra_agents(config_id=config_id, persid=agent.agenda.persid,
                                                             agent=agent)
                df = to_df(agent=agent, agent_id=agent_id, poi_route=poi_route, survey_encoding=self.survey_encoding)

                df.to_csv('dfg.csv', index=False)
                self.database.append_astra_routes_checkin(df=df)

                yield agent, df
