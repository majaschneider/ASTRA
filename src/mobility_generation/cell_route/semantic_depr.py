import numpy as np
import attr
import ray
from typing import Type

import pandas as pd

from src.helper.helper import CellEmptyException
from src.config import RouteCreationConfig
from src.survey_processing.loading.travel_survey_encoding import SurveyEncoding
from src.tessellation.tessellation.tessellation import Tessellation
from src.mobility_generation.data_classes.agent import Agent
from src.mobility_generation.data_classes.route import RouteElement
import src.mobility_generation.utils.scoring as scoring


@attr.define(slots=True, frozen=True)
class SemanticDEPR:
    """class to predict the next choice of a cell."""

    # epr parameters
    rho: float = attr.field()
    gamma: float = attr.field()

    # recency parameter
    alpha: float = attr.field()
    eta: float = attr.field()

    # option to embed persona features next to the activity segments/travel activities
    embed_persona_features: bool = attr.field()

    allow_returns: bool = attr.field()

    # score weights
    weight_explore_poi_type_frequency: float = attr.field()
    weight_explore_semantic_similarity: float = attr.field()
    weight_explore_gravity: float = attr.field()

    # preferential return score weights
    weight_return_frequency: float = attr.field()
    weight_return_recency: float = attr.field()
    weight_return_semantic_similarity: float = attr.field()

    # minimum similarity score between activities and visited locations to allow a return
    return_min_similarity_score: float = attr.field()

    # number of highest ranking pois to choose from
    top_n: int = attr.field()

    # random number generator for probabilistic selection
    rng: np.random.Generator = attr.field()

    # max travel time percentage error where location is still considered reachable
    max_travel_time_percentage_error: float = attr.field(validator=(attr.validators.lt(1.0), attr.validators.gt(0.0)))

    survey_encoding: SurveyEncoding = attr.field()

    @classmethod
    def from_config(
        cls,
        survey_encoding: SurveyEncoding,
        route_creation_config: RouteCreationConfig,
        rng: np.random.Generator,
    ) -> "SemanticDEPR":
        """
        create decision class from route creation config
        requires embedding collection as kwarg
        """
        return cls(
            rho=route_creation_config.rho,
            gamma=route_creation_config.gamma,
            alpha=route_creation_config.alpha,
            eta=route_creation_config.eta,
            embed_persona_features=route_creation_config.embed_persona_features,
            allow_returns=route_creation_config.allow_returns,
            rng=rng,
            max_travel_time_percentage_error=route_creation_config.max_travel_percentage_error,
            weight_explore_poi_type_frequency=route_creation_config.weight_explore_poi_type_frequency,
            weight_explore_semantic_similarity=route_creation_config.weight_explore_semantic_similarity,
            weight_explore_gravity=route_creation_config.weight_explore_gravity,
            weight_return_frequency=route_creation_config.weight_return_frequency,
            weight_return_recency=route_creation_config.weight_return_recency,
            weight_return_semantic_similarity=route_creation_config.weight_return_semantic_similarity,
            return_min_similarity_score=route_creation_config.return_min_similarity_score,
            top_n=route_creation_config.top_n,
            survey_encoding=survey_encoding
        )

    @classmethod
    def get_agent_type(cls) -> Type[Agent]:
        """
        get type of agent used in the ClusterChoice class
        """
        return Agent

    def _preferential_return(self, agent: Agent, tessellation: Tessellation) -> RouteElement:
        """
        Preferential return step of EPR model. Recency of location visits is considered according to
        Barbosa et al. (2015).

        See:
        Barbosa, Hugo, Fernando B De Lima-Neto, Alexandre Evsukoff, und Ronaldo Menezes.
        „The Effect of Recency to Human Mobility“. EPJ Data Science 4, Nr. 1 (Dezember 2015): 21.
        https://doi.org/10.1140/epjds/s13688-015-0059-8.
        """
        # get previously visited locations except for current location
        visited_locations_unmasked = np.asarray(tuple(zip(agent.history.cell_idxs, agent.history.embedding_ids)))

        # filter out last visit and visits that have the same cell_id and poi as the agent's last visited location
        mask = ~agent.history.equals_last
        visited_locations = visited_locations_unmasked[mask]

        # if no locations available to return to, change to exploration mode instead
        if len(visited_locations) == 0:
            return self.explore(agent=agent, tessellation=tessellation)

        # todo: make array in class and remove here
        visited_cell_ids = np.asarray(agent.history.cell_idxs)[mask]
        visited_pois = np.asarray(agent.history.embedding_ids)[mask]
        unique_visited_pois = np.unique(visited_pois)
        unique_visited_locations = np.unique(visited_locations, axis=0)

        # Calculate a frequency rank for each visited cell id indicating how often it has been visited by the agent in
        # descending order
        freq_locations, freq_rank = scoring.calculate_frequency_ranks(visited_locations)

        # Calculate a recency rank for each visited location indicating how recent it was that a location was visited
        # from least recent to most recent
        rec_locations, rec_rank = scoring.recency_rank(visited_locations)

        assert (np.all(freq_locations == rec_locations) and np.all(freq_locations == unique_visited_locations))

        # todo: read paper and document this
        # calculate likelihood of choosing one of the unique cell ids based on their frequency and recency in the
        # agent's history
        frequency_score = self.alpha * np.power(freq_rank, -1 - self.gamma)
        recency_score = (1 - self.alpha) * np.power(rec_rank, -self.eta)

        frequency_scores_dict = dict(zip([str(loc) for loc in freq_locations], frequency_score))
        recency_scores_dict = dict(zip([str(loc) for loc in rec_locations], recency_score))
        # assign scores to each location in the agent's history
        frequency_score_per_visited_poi = np.asarray([frequency_scores_dict[key]
                                                      for key in [str(loc) for loc in unique_visited_locations]])
        recency_score_per_visited_poi = np.asarray([recency_scores_dict[key]
                                                    for key in [str(loc) for loc in unique_visited_locations]])

        persona_features = self._create_embedding_prefix(agent.agenda.persona_features)

        travel_mode = ray.get(
            tessellation.embedding_collection.get_travel_mode.remote(
                encoded_travel_mode=agent.current_segment.travel_mode,
                prefix=persona_features
            )
        )

        # get the codes and durations of the agent's activities that were executed in the segment that is currently
        # being processed
        activity_codes = np.fromiter(agent.current_segment.activity_durations.keys(), dtype=int)
        activity_durations = np.fromiter(agent.current_segment.activity_durations.values(), dtype=float)

        # form a search query from the agent's persona features (optional) and the current segment's activities and
        # calculate a similarity score between the search query and each visited poi in the agent's history
        similarity_scores_per_poi, similar_pois = ray.get(
            tessellation.embedding_collection.find_most_similar_pois.remote(
                encoded_activities=activity_codes,
                weights=activity_durations,
                prefix=persona_features,
                ids_to_search=unique_visited_pois,
            )
        )

        # if similarity between activities and available locations is too low, choose exploration mode instead
        if np.max(similarity_scores_per_poi) < self.return_min_similarity_score:
            print('return denied', np.max(similarity_scores_per_poi))
            return self.explore(agent=agent, tessellation=tessellation)
        else:
            print('return allowed', np.max(similarity_scores_per_poi))

        # make a dict indicating the similarity of a unique poi with its embedding id as key
        similarity_scores_dict = dict(zip(similar_pois, similarity_scores_per_poi))
        del similar_pois, similarity_scores_per_poi

        # look up similarities and assign it to each of the visited POIs (sorted by the visited POIs)
        similarity_scores_per_visited_poi = np.asarray([similarity_scores_dict[poi]
                                                        for _, poi in unique_visited_locations])


        # make sure no negative similarity values are present
        similarity_scores_per_visited_poi = np.asarray([0.001 + value - np.min(similarity_scores_per_visited_poi)
                                             for value in similarity_scores_per_visited_poi])

        weights = [self.weight_return_frequency, self.weight_return_recency, self.weight_return_semantic_similarity]
        scores = [frequency_score_per_visited_poi, recency_score_per_visited_poi, similarity_scores_per_visited_poi]

        weighted_score_per_visited_poi = scoring.calculate_normalized_weighted_scores(scores, weights)

        chosen_id = self.select_item(scores=weighted_score_per_visited_poi, top_n=self.top_n)
        chosen_poi = int(visited_pois[chosen_id])
        chosen_cell_id = int(visited_cell_ids[chosen_id])
        # todo: impact of duration at poi on semantics (as a score)

        chosen_id_in_visited_elements = 0
        for i, (cell_id, embedding_id) in enumerate(visited_locations_unmasked):
            if cell_id == chosen_cell_id and embedding_id == chosen_poi:
                chosen_id_in_visited_elements = i

        route_element = RouteElement(
            cell_idx=chosen_cell_id,
            embedding_idx=chosen_poi,
            segment_range=agent.current_segment.segment_range,
            travel_mode=travel_mode,
            idx_of_first_visit=chosen_id_in_visited_elements
        )

        return route_element

    def _create_embedding_prefix(self, persona_features: dict) -> str:
        """
        If the option is set in the configuration, create a prefix from an agent's persona features. The prefix can be
        used as additional search term in a sentence embedding.
        """
        prefix = ""
        if self.embed_persona_features:
            prefix = ''

            gender_decoding = {1: 'Male', 2: 'Female'}
            gender_code = persona_features['SEX']
            if gender_code in gender_decoding.keys():
                gender = gender_decoding[gender_code]
                prefix += f"{gender}"

            prefix += f" person of age {persona_features['AGE']}"
        return prefix

    def is_exploration_mode_chosen(self, current_route_length):
        if current_route_length == 0:
            return False

        exploration_probability = self.calculate_exploration_probability(num_visited_locations=current_route_length)

        is_exploration_mode = self.rng.choice(
            a=(True, False),
            p=(exploration_probability, 1 - exploration_probability)
        )

        return is_exploration_mode

    def calculate_next_route_element(self, agent: Agent, tessellation: Tessellation) -> RouteElement:
        """
        Choose the cell in the tesselation that the agent will visit next in their route. The cell is chosen based on
        the exploration and preferential return (EPR) model. If the agent's route is still empty, choose the agent's
        starting cell. Returns to the previously visited cell are not allowed.
        """
        # disallow returns for the first two stops in the route
        calculate_first_or_second_stop = len(agent.history) < 2

        # choose exploration mode based on EPR model likelihood
        is_exploration_mode_chosen = self.is_exploration_mode_chosen(agent.history.num_unique_locations)
        # todo: first segment should be home address
        # todo: add segment info in agenda, if last segment is reached, add first segment's location if it is not the same or not a home address

        if calculate_first_or_second_stop or is_exploration_mode_chosen or not self.allow_returns:
            return self.explore(agent=agent, tessellation=tessellation)
        else:
            return self._preferential_return(agent=agent, tessellation=tessellation)

    def explore(self, agent: Agent, tessellation: Tessellation) -> RouteElement:
        """
        Exploration step of EPR model. A reachable cell in the tesselation is chosen based on the allowed travel time
        error between available POIs and the agent's activity. If no cells are reachable within the given travel time,
        relax the allowed error to find more reachable cells.
        Semantic similarity is calculated between the current agenda segment's activities, the agent's persona features
        (optionally), the frequency of a POI's type, and a gravity score between the agent's last visited cell and the
        potential next cell. If the exploration is aiming to find the start cell of the agent, gravity score is ignored.
        """
        calculate_first_route_element = len(agent.history) == 0

        # get the agents persona features and process them to the required format for an embedding
        persona_features = self._create_embedding_prefix(persona_features=agent.agenda.persona_features)

        # get selectable pois
        if calculate_first_route_element:
            travel_mode = None

            # get POIs from the agent's start cell
            poi_set = tessellation.get_poi_set(agent.start_cell_idx)
            cell_idxs = [agent.start_cell_idx] * len(poi_set.embedding_ids)
            pois = poi_set.embedding_ids
            poi_type_frequencies = poi_set.poi_sentence_frequencies
            if len(pois) == 0:
                raise CellEmptyException('Error in exploration: Start cell does not contain any POIs to choose from.')
        else:
            # get most likely travel mode used to get to the next segment
            travel_mode = ray.get(
                tessellation.embedding_collection.get_travel_mode.remote(
                    encoded_travel_mode=agent.current_segment.travel_mode,
                    prefix=persona_features
                )
            )

            # get POIs from all reachable cells that have not yet been visited by the agent
            # allow a travel time error to reach further cells if there are no POIs in the reachable cells
            pois_df = self.calculate_selectable_pois(agent=agent, tessellation=tessellation, travel_mode=travel_mode)
            cell_idxs = np.asarray(pois_df['cell_idx'])
            pois = np.asarray(pois_df['embedding_id'])
            poi_type_frequencies = np.asarray(pois_df['poi_type_frequency_in_cell'])

        # get the codes and durations of the agent's activities that were executed in the segment that is currently
        # being processed
        activity_codes = np.fromiter(agent.current_segment.activity_durations.keys(), dtype=int)
        activity_durations = np.fromiter(agent.current_segment.activity_durations.values(), dtype=float)
        print('current segment\'s activities', self.survey_encoding.decode_activities(activity_codes))

        unique_pois = np.unique(pois)

        # form a search query from the agent's persona features (optional) and the current segment's activities and
        # calculate a similarity score between the search query and each unique poi in this poi set
        similarity_scores_per_poi, similar_pois = ray.get(
            tessellation.embedding_collection.find_most_similar_pois.remote(
                encoded_activities=activity_codes,
                weights=activity_durations,
                prefix=persona_features,
                ids_to_search=unique_pois,
            )
        )

        # make a dict indicating the similarity of a unique poi with its embedding id as key
        similarity_scores_dict = dict(zip(similar_pois, similarity_scores_per_poi))
        del similar_pois, similarity_scores_per_poi

        # look up similarities and assign it to each of the poi sets' POIs (sorted by the poi sets' POIs)
        similarity_scores_per_poi = np.fromiter(
            (similarity_scores_dict[embedding_id] for embedding_id in pois),
            dtype=float
        )

        weights = [self.weight_explore_poi_type_frequency, self.weight_explore_semantic_similarity,
                   self.weight_explore_gravity]

        # calculate and normalize scores and calculate a weighted score
        if calculate_first_route_element:
            gravity_scores = np.ones(len(poi_type_frequencies))
            # todo: keep ratio
            for i in range(len(weights) - 1):
                weights[i] += self.weight_explore_gravity / (len(weights) - 1)
            weights[-1] = 0
        else:
            # todo: check if cell_ids is correct
            gravity_scores = tessellation.od_gravity_scores[agent.history.last_cell_idx, cell_idxs].ravel()

        # reduce impact of outliers by applying log function
        poi_type_frequencies = np.log2(poi_type_frequencies + 1)
        # make sure no negative similarity values are present
        similarity_scores_per_poi = np.asarray(
            [value - np.min(similarity_scores_per_poi) for value in similarity_scores_per_poi]
        )

        scores = [poi_type_frequencies, similarity_scores_per_poi, gravity_scores]
        weighted_score_per_poi = scoring.calculate_normalized_weighted_scores(scores, weights)

        chosen_id = self.select_item(scores=weighted_score_per_poi, top_n=self.top_n)
        chosen_poi = int(pois[chosen_id])
        chosen_cell_idx = cell_idxs[chosen_id]
        # todo: impact of duration at poi on semantics (as a score)

        route_element = RouteElement(
            cell_idx=chosen_cell_idx,
            embedding_idx=chosen_poi,
            segment_range=agent.current_segment.segment_range,
            travel_mode=travel_mode
        )
        print('chosen cell_idx:', chosen_cell_idx, 'poi embedding id:', chosen_poi)

        return route_element

    def select_item(self, scores: np.ndarray, top_n: int) -> int:
        """
        Select an item from the top-n highest scoring items proportionally to their score. Return its index in
        scores.
        """
        nr_items = len(scores)

        if top_n is None or top_n < 1:
            top_n = nr_items

        top_n = np.min([top_n, nr_items])

        # get ids of top-n highest scoring items
        ids_and_scores = sorted(enumerate(scores), key=lambda x:x[1], reverse=True)[:top_n]
        selected_n_scores = np.asarray([score for _, score in ids_and_scores])

        # normalize scores
        selected_n_scores = selected_n_scores / np.sum(selected_n_scores)

        # select an item according to its score
        idx = self.rng.choice(a=range(top_n), p=selected_n_scores)
        chosen_id = ids_and_scores[idx][0]

        return chosen_id

    def calculate_selectable_pois(self, agent: Agent, tessellation: Tessellation, travel_mode: str):
        # get POIs that have already been visited by the agent
        visited_pois_df = pd.DataFrame(zip(agent.history.cell_idxs, agent.history.embedding_ids),
                                       columns=['cell_idx', 'embedding_id'])
        visited_pois_df = visited_pois_df.groupby(['cell_idx', 'embedding_id']).agg(
            visit_count=('embedding_id', 'count')
        ).reset_index()

        # if there are no POIs in the reachable cells, relax travel time until cells with POIs are found
        travel_time_relaxation_minutes = 5
        travel_time_minutes = agent.current_segment.travel_time_mins
        travel_time_max = 24 * 60
        # current_activities = agent.current_segment.activity_durations.keys()

        for travel_time_error in range(0, travel_time_max + 1, travel_time_relaxation_minutes):
            # get reachable cells
            reachable_cell_idxs = tessellation.get_reachable_cell_idxs(
                origin_cell_idx=agent.history.last_cell_idx,
                travel_mode=travel_mode,
                travel_time_minutes=travel_time_minutes,
                max_error=travel_time_error,
            )

            # get reachable POIs
            poi_sets = [(cell_id, tessellation.get_poi_set(idx=cell_id)) for cell_id in reachable_cell_idxs]
            nr_pois = np.sum([poi_set.get_number_of_pois() for _, poi_set in poi_sets])

            if nr_pois > 0:
                # transform POIs to DataFrame
                pois = np.concatenate(tuple(
                    (np.asarray(tuple(zip(
                        [cell_idx] * len(poi_set.embedding_ids),
                        poi_set.embedding_ids,
                        poi_set.poi_sentence_frequencies,
                    )))
                        for cell_idx, poi_set in poi_sets)
                ))
                pois_df = pd.DataFrame(pois, columns=['cell_idx', 'embedding_id', 'poi_type_frequency_in_cell'])

                # exclude POIs that have been visited from reachable POIs
                pois_df = pois_df.merge(visited_pois_df, how='left', on=['cell_idx', 'embedding_id']).fillna(0)
                pois_df = pois_df[pois_df['visit_count'] < pois_df['poi_type_frequency_in_cell']]

                # exit loop once cells with valid POIs found
                if len(pois_df) > 0:
                    return pois_df

        raise IOError('There are no POIs to select in the simulation area.')

    def calculate_exploration_probability(self, num_visited_locations: int) -> float:
        """
        Calculate probability of exploration according to exploration and preferential return model. With probability
            p = rho * N ** (−gamma) where
            N is the number of distinct visited locations in an agent's route, and
            rho = 0.6, gamma = 0.21 are constants,
        an agent chooses to explore a new location, otherwise returns to a previously visited location.

        Pappalardo, Luca, Filippo Simini, Salvatore Rinzivillo, Dino Pedreschi, Fosca Giannotti, und Albert-László Barabási.
        „Returners and Explorers Dichotomy in Human Mobility“.
        Nature Communications 6, Nr. 1 (8. September 2015): 8166. https://doi.org/10.1038/ncomms9166.
        """
        exploration_probability = self.rho * num_visited_locations ** (-self.gamma)

        return exploration_probability