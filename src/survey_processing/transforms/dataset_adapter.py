import numpy as np
import pandas as pd
from gluonts.dataset.common import (
    TrainDatasets,
    load_datasets,
    CategoricalFeatureInfo,
    BasicFeatureInfo,
    MetaData,
)
from pathlib import Path
from dataclasses import replace
from itertools import groupby
from gluonts.dataset.arrow import ParquetWriter
from datetime import timedelta, datetime
from typing import Iterator, NamedTuple, Iterable, Any
from gluonts.dataset.pandas import PandasDataset
from gluonts.model import Forecast
from sklearn.model_selection import train_test_split

from src.config import SurveyProcessingConfig
from src.survey_processing.dataclasses.agenda import Agenda
from src.tessellation.data_classes.demographic import Demographic
from src.survey_processing.transforms.scaler import Scaler
from src.survey_processing.loading.travel_survey_encoding import (
    SurveyEncoding,
)
from src.survey_processing.loading.survey_loading import (
    SurveyLoader,
)


class DatasetAdapter(NamedTuple):
    """
    class for constructing sequence datasets from agenda data
    """

    scaler: Scaler
    survey_encoding: SurveyEncoding

    persona_feature_names: list[str]
    categorical_persona_feature_names: list[str]
    real_persona_feature_names: list[str]
    prepend_persona_features: bool

    time_step_interval: int

    test_size: float
    seed: int
    rng: np.random.Generator

    @classmethod
    def from_config(
        cls,
        scaler: Scaler,
        survey_encoding: SurveyEncoding,
        survey_processing_config: SurveyProcessingConfig,
        seed: int,
        rng: np.random.Generator,
    ) -> "DatasetAdapter":
        return cls(
            scaler=scaler,
            survey_encoding=survey_encoding,
            persona_feature_names=survey_processing_config.persona_feature_names,
            categorical_persona_feature_names=survey_processing_config.categorical_persona_feature_names,
            real_persona_feature_names=survey_processing_config.real_persona_feature_names,
            prepend_persona_features=survey_processing_config.prepend_persona_features,
            time_step_interval=survey_processing_config.time_step_interval,
            test_size=survey_processing_config.test_size,
            seed=seed,
            rng=rng,
        )

    def create_dataset_from_agendas(
        self, agendas: Iterable[tuple[Agenda, int]]
    ) -> TrainDatasets:
        """
        aggregate agendas in sequence model training and test datasets
        """
        activities, persona_features, prefix_timestamps = [], [], []
        prediction_length = -1

        # aggregate data vectors from every agenda
        for agenda, freq in agendas:
            # determine maximum agenda length. Is then further used as prediction length
            agenda_len = int(agenda.episode_ranges[-1][1])
            if agenda_len > prediction_length:
                prediction_length = agenda_len

            df = agenda.to_df()
            df["target"] = self.scaler.agenda_scaler.transform(
                df["target"].to_numpy().reshape((-1, 1))
            )

            prefix = pd.date_range(
                start=agenda.starting_timestamp
                - timedelta(
                    minutes=agenda.time_step_interval
                    * len(agenda.persona_features) ** int(self.prepend_persona_features)
                ),
                freq="%sT" % agenda.time_step_interval,
                end=agenda.starting_timestamp
                - timedelta(minutes=agenda.time_step_interval),
            )

            for _ in range(freq):
                copy_df = df.copy()
                copy_df["item_id"] = len(activities)
                activities.append(copy_df)
                prefix_timestamps.append(prefix)
                persona_features.append(agenda.persona_features)

        # scale persona features
        persona_features = self.scaler.transform_persona_features(
            pd.DataFrame(data=persona_features)[self.persona_feature_names]
        )

        # convert categorical persona features to categorical data type
        persona_features = self.scaler.to_categorical(
            persona_features, self.categorical_persona_feature_names
        )

        metadata = MetaData(
            freq="%sT" % self.time_step_interval,
            prediction_length=prediction_length,
            feat_dynamic_real=[],
            feat_dynamic_stat=[],
            feat_static_cat=[
                CategoricalFeatureInfo(
                    name=name,
                    cardinality=str(persona_features[name].nunique()),
                )
                for name in self.categorical_persona_feature_names
            ],
            feat_static_real=[
                BasicFeatureInfo(name=name) for name in self.real_persona_feature_names
            ],
        )

        # train test split agendas
        train_indices, test_indices = train_test_split(
            np.arange(persona_features.shape[0]),
            test_size=self.test_size,
            random_state=self.seed,
        )

        train_static_features = persona_features.iloc[train_indices]
        test_static_features = persona_features.iloc[test_indices]

        train_dfs = (
            pd.concat((prefix, activity))
            for prefix, activity in zip(
                _create_prefixes(
                    persona_features=train_static_features,
                    timestamps=[prefix_timestamps[i] for i in train_indices],
                    prepend_persona_features=self.prepend_persona_features,
                ),
                (activities[i] for i in train_indices),
            )
        )

        test_dfs = (
            pd.concat((prefix, activity))
            for prefix, activity in zip(
                _create_prefixes(
                    persona_features=test_static_features,
                    timestamps=[prefix_timestamps[i] for i in test_indices],
                    prepend_persona_features=self.prepend_persona_features,
                ),
                (activities[i] for i in test_indices),
            )
        )

        dataset = TrainDatasets(
            metadata=metadata,
            train=PandasDataset(
                dataframes=dict(zip(train_indices, train_dfs)),
                static_features=train_static_features,
                freq=metadata.freq,
            ),
            test=PandasDataset(
                dataframes=dict(zip(test_indices, test_dfs)),
                static_features=test_static_features,
                freq=metadata.freq,
            ),
        )

        return dataset

    def create_dataset_from_demographic(
        self,
        demographic: Demographic,
        starting_timestamp: datetime,
        starting_timestamp_std: timedelta,
    ) -> tuple[PandasDataset, pd.DataFrame]:
        """
        create prefix dataset for given demographic and starting timestamp.
        """

        # draw prefixes from the demographics prefix distribution
        prefix_distribution = demographic.select_agents_persona_features(rng=self.rng)

        persona_feature_df = prefix_distribution[self.persona_feature_names]

        # encode persona features
        encoded_persona_feature_df = persona_feature_df.copy()
        encoded_persona_feature_df.loc[:, self.persona_feature_names] = (
            encoded_persona_feature_df.apply(
                lambda x: pd.Series(
                    self.survey_encoding.encode_persona_features(x.to_dict())
                ),
                axis=1,
            )
        )

        # scale persona features
        encoded_persona_feature_df.loc[:, self.persona_feature_names] = (
            self.scaler.transform_persona_features(encoded_persona_feature_df)
        )

        # create time series df
        time_series = pd.DataFrame(
            data={
                "item_id": np.arange(demographic.nr_simulated_agents),
                "timestamp": _sample_timestamps(
                    base_timestamp=starting_timestamp,
                    time_std=starting_timestamp_std,
                    num_timestamps=demographic.nr_simulated_agents,
                    rng=self.rng,
                ),
            },
        )
        time_series["timestamp"] = time_series["timestamp"].apply(
            lambda x: pd.date_range(
                start=x
                - timedelta(
                    minutes=self.time_step_interval
                    * (
                        len(self.persona_feature_names)
                        ** int(self.prepend_persona_features)
                        - 1
                    ),
                ),
                end=x,
                freq=f"{self.time_step_interval}T",
            ).tolist(),
        )

        encoded_persona_feature_df = encoded_persona_feature_df.assign(
            item_id=np.arange(demographic.nr_simulated_agents)
        )
        # if option is set use persona features as time series prefixes, else na prefix
        if self.prepend_persona_features:
            time_series = time_series.merge(
                encoded_persona_feature_df, on="item_id", how="left"
            )
            time_series["target"] = time_series[self.persona_feature_names].apply(
                lambda x: x.values, axis=1
            )
            time_series.drop(
                columns=self.persona_feature_names,
                inplace=True,
            )

            time_series = time_series.explode(
                ["timestamp", "target"], ignore_index=True
            ).reset_index(drop=True)
        else:
            time_series["target"] = np.nan
            time_series["timestamp"] = time_series["timestamp"].apply(lambda x: x[0])

        # reindex persona features
        encoded_persona_feature_df.set_index("item_id", inplace=True, drop=True)

        # convert categorical features to categories
        encoded_persona_feature_df = self.scaler.to_categorical(
            encoded_persona_feature_df,
            categorical_feature_names=self.categorical_persona_feature_names,
        )

        # create time series dataset
        dataset = PandasDataset.from_long_dataframe(
            dataframe=time_series,
            item_id="item_id",
            timestamp="timestamp",
            static_features=encoded_persona_feature_df,
            freq=f"{self.time_step_interval}T",
            future_length=0,
        )

        return dataset, persona_feature_df

    def get_all_dataset_values(self) -> np.ndarray:
        """
        get all values a sequence in a sequence dataset can take
        """
        return self.scaler.agenda_scaler.transform(
            self.survey_encoding.all_encoded().reshape(-1, 1)
        )

    def inverse_transform_forecast(
        self, forecast: Forecast, persona_features: dict[str, Any]
    ) -> Agenda:
        """
        inverse transform sequence model forecast to agenda
        """
        # sample agenda from output
        series = self.rng.choice(forecast.samples)

        # inverse transform scaled values
        series = self.scaler.agenda_scaler.inverse_transform(
            series.reshape((-1, 1))
        ).squeeze(axis=1)
        # transform to valid values
        series = map(self.survey_encoding.get_valid_agenda_element, series)

        # create activity episode ranges
        episode_ranges, activities, travel = [], [], []
        position = 0
        for i, (element, group) in enumerate(groupby(series)):
            new_position = position + len(tuple(group))
            episode_ranges.append((position, new_position))
            activities.append(element)

            if (
                element in self.survey_encoding.travel_decoding.keys()
                or element in self.survey_encoding.travel_activities
            ):
                travel.append(i)

            position = new_position

        episode_ranges = np.asarray(episode_ranges, dtype=int)
        activities = np.asarray(activities, dtype=int)
        travel = np.asarray(travel, dtype=int)

        # create Agenda
        agenda = Agenda(
            activities=activities,
            travel=travel,
            episode_ranges=episode_ranges,
            persona_features=persona_features,
            starting_timestamp=forecast.start_date.start_time,
            time_step_interval=self.time_step_interval,
        )
        agenda.preprocess_travel()

        return agenda


def load_or_create_dataset(
    path_processed_dataset: Path,
    force_reload_dataset: bool,
    survey_paths: Iterable[Path],
    loading_modules: Iterable[SurveyLoader] | SurveyLoader,
    dataset_factory: DatasetAdapter,
) -> TrainDatasets:
    """
    load or create gluonts supported travel survey dataset
    """

    # create paths and make directories for saving/loading the datasets
    path_processed_dataset.mkdir(parents=True, exist_ok=True)
    path_train = path_processed_dataset / "train"
    path_test = path_processed_dataset / "test"

    if (
        any(path_processed_dataset.iterdir())
        and path_train.exists()
        and not force_reload_dataset
    ):
        print("Loading saved dataset ...")
        return load_datasets(
            metadata=path_processed_dataset,
            train=path_train,
            test=path_test if path_test.exists() else None,
        )
    else:
        # create dataset_factory to aggregate data from all loading modules
        if isinstance(loading_modules, Iterable):
            survey_agendas = (
                loading_module.load_from_paths(path=path)
                for loading_module, path in zip(loading_modules, survey_paths)
            )
        elif isinstance(loading_modules, SurveyLoader):
            survey_agendas = (
                loading_modules.load_from_paths(path) for path in survey_paths
            )
        else:
            raise ValueError(
                "Error when loading dataset, provided loading_modules is not an iterable or a SurveyLoader"
            )
        agendas = (agenda for survey in survey_agendas for agenda in survey)
        # create one dataset from all agendas
        dataset = dataset_factory.create_dataset_from_agendas(agendas=agendas)

        # save dataset and scaler at specified locations
        dataset.save(
            path_str=str(path_processed_dataset),
            writer=ParquetWriter(),
            overwrite=True,
        )
        dataset_factory.scaler.save()
        return dataset


def _sample_timestamps(
    base_timestamp: datetime,
    time_std: timedelta,
    num_timestamps: int,
    rng: np.random.Generator,
) -> Iterator[datetime]:
    """
    draw a sample of timestamps from a normal distribution around the given base timestamp.
    """
    assert num_timestamps > 0
    time_std = round(time_std.total_seconds() / 60)
    for deviation in rng.normal(loc=0, scale=time_std, size=num_timestamps):
        yield base_timestamp + timedelta(minutes=deviation)


def _create_prefixes(
    persona_features: pd.DataFrame,
    timestamps: list[pd.DatetimeIndex],
    prepend_persona_features: bool,
) -> Iterator[pd.DataFrame]:
    if prepend_persona_features:
        persona_features = pd.DataFrame(persona_features).to_numpy()
        for i in range(persona_features.shape[0]):
            df = pd.DataFrame(
                data={
                    "target": persona_features[i, :],
                },
                index=timestamps[i],
            )
            yield df
    else:
        for i, timestamp in enumerate(timestamps):
            df = pd.DataFrame(data={"target": np.nan}, index=timestamp)
            yield df
