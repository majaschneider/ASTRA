import ray
import modin.pandas as modin
import numpy as np
from abc import ABC, abstractmethod
from calendar import monthrange
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Iterable, Any
import os, psutil

from src.config import SurveyProcessingConfig
from src.survey_processing.dataclasses.agenda import Agenda
from src.survey_processing.utils.utils_time_use_processing import (
    pick_random_cday,
    split_into_windows,
)
from src.survey_processing.loading.travel_survey_encoding import (
    SurveyEncoding,
)

# os.environ["MODIN_CPUS"] = "2"

class SurveyLoader(ABC):
    """
    Abstract base class for loading survey data into agendas
    """

    @abstractmethod
    def load_from_paths(self, path: Path, nr_agents: int | None = None) -> Iterator[tuple[Agenda, int]]:
        """
        create encoded agendas from data in path
        """
        ...


class MTUSLoader(SurveyLoader):
    """
    Load MTUS survey data into agendas
    """

    __rng: np.random.Generator

    __time_step_interval: int
    __sample_size: int
    __dummy_date: datetime
    __persona_feature_names: list[str]
    __prepend_persona_features: bool

    __survey_encoding: SurveyEncoding

    def __init__(
        self,
        config: SurveyProcessingConfig,
        encoding: SurveyEncoding,
        seed: int,
    ):
        self.__time_step_interval = config.time_step_interval
        self.__sample_size = config.travel_survey_sample_size
        self.__dummy_date = config.dummy_date
        self.__persona_feature_names = config.persona_feature_names
        self.__prepend_persona_features = config.prepend_persona_features

        self.__survey_encoding = encoding
        self.__rng = np.random.default_rng(seed)

    def load_from_paths(self, path: Path, nr_agents: int | None = None) -> Iterator[tuple[Agenda, int]]:
        """
        Load survey data from the given path and generate agendas for agents.
        :param path: Path to the survey data file.
        :param nr_agents: Number of agents to load.
        :return: Iterator of Agenda objects.
        """
        activity_df = self.load_df(path, nr_agents)

        # Group the data by agenda identifiers
        activity_groups = activity_df.groupby(["IDENT", "SAMPLE"], sort=False, dropna=False)

        # aggregate agenda data to agenda
        agendas_and_frequency = activity_groups.apply(self._aggregate_into_agenda)

        # Yield each agenda
        for agenda, frequency in agendas_and_frequency:
            agenda.preprocess_travel()

            yield tuple([agenda, frequency])

    def load_df(self, path: Path, nr_agents: int | None = None) -> modin.DataFrame:
        """
        Creates a DataFrame from the dataset at path. Takes into account a weighting in the dataset, to guarantee an
        even distribution of agents over the day of the week and age and sex groups. Further cleaning is applied to
        remove missing or invalid data.
        """
        assert path.exists()
        print('Loading dataframe')

        # Load the CSV file using modin with optimized data types.
        activity_df: modin.DataFrame = modin.read_csv(
            path,
            dtype={  # Specify data types for each column to optimize memory usage.
                "SAMPLE": str,
                "IDENT": "Int32",
                "COUNTRY": str,
                "HLDID": "Int64",
                "PERSID": "Int64",
                "DIARY": "Int8",
                "YEAR": "Int16",
                "DIARYID": "Int8",
                "DAY": "Int8",
                "MONTH": "Int8",
                "VEHICLE": "Int8",
                "AGE": "Int8",
                "SEX": "Int8",
                "CITIZEN": "Int8",
                "EMPSTAT": "Int8",
                "STUDENT": "Int8",
                "INCOME": "Int8",
                "PROPWT": "float16",
                "CDAY": "Int8",
                "EPNUM": "Int8",
                "CLOCKST": "float16",
                "START": "Int16",
                "END": "Int16",
                "TIME": "Int16",
                "MAIN": "Int8",
                "ELOC": "Int8",
                "INOUT": "Int8",
                "MTRAV": "Int8",
                "CIVSTAT": "Int8",
                "EDTRY": "Int8",
                "CARER": "Int8",
                "MIGRANTD": "Int8",
                "MIGRANTM": "Int8",
                "MIGRANTF": "Int8",
                "NCHILD": "Int8",
                "URBAN": "Int8",
                "SINGPAR": "Int8",
                "AGEKID": "Int8",
                "HEALTH": "Int8",
                "RUSHED": "Int8",
                "DISAB": "Int8",
                "ALONE": "Int8",
                "CHILD": "Int8",
                "SPPART": "Int8",
                "OAD": "Int8",
            },
        )
        # filter for valid samples from the US
        activity_df = activity_df[activity_df["COUNTRY"] == "US"]
        activity_df = activity_df[activity_df["PROPWT"] > 0]

        # project to relevant columns
        activity_df.dropna(subset=["IDENT", "PERSID", "DIARY", "SAMPLE", "MAIN"], inplace=True)
        activity_df.reset_index(drop=True, inplace=True)

        # filter out agendas with unspecified activities
        invalid_activity_indices = activity_df["MAIN"].isin((69, 99))

        unique_agenda_ids = activity_df[["IDENT", "SAMPLE", "PROPWT"]].drop_duplicates()

        invalid_rows = activity_df.loc[invalid_activity_indices, ["IDENT", "SAMPLE"]]
        if not invalid_rows.empty:
            invalid_rows.drop_duplicates(inplace=True)
            invalid_rows["invalid"] = True

            unique_agenda_ids = unique_agenda_ids.merge(
                invalid_rows,
                on=["IDENT", "SAMPLE"],
                how="left",
            )

            unique_agenda_ids = unique_agenda_ids[modin.isna(unique_agenda_ids["invalid"])]
            unique_agenda_ids.drop(columns=["invalid"], inplace=True)

        # sample agendas proportionally to their weight: "PROPWT reports the proposed weight used when population and
        # day combined weights need to be rescaled"
        weights = unique_agenda_ids["PROPWT"].to_numpy(dtype=float)
        unique_agenda_ids.drop(columns="PROPWT", inplace=True)
        sum_weights = weights.sum()

        if self.__sample_size == -1:
            self.__sample_size = np.floor(sum_weights)
        self.__sample_size = np.min([self.__sample_size, len(unique_agenda_ids)])

        agent_ids_weighted = self.__sample_size * weights / sum_weights
        agents_ids_chosen = np.floor(agent_ids_weighted)
        weight_remainders = agent_ids_weighted - agents_ids_chosen
        ids_of_descending_weight = np.argsort(-weight_remainders)
        remaining_weight_to_choose_agents = int(np.floor(weight_remainders.sum()))
        ids_to_choose = ids_of_descending_weight[:remaining_weight_to_choose_agents]
        agents_ids_chosen[ids_to_choose] += 1

        unique_agenda_ids["freq"] = agents_ids_chosen
        unique_agenda_ids = unique_agenda_ids[unique_agenda_ids["freq"] > 0]
        activity_df = activity_df.merge(
            unique_agenda_ids,
            on=["IDENT", "SAMPLE"],
            how="inner",
        )

        # Limit the dataset to the specified number of agents
        if nr_agents is not None and nr_agents > 0:
            identifying_columns = ["IDENT", "PERSID", "DIARY", "SAMPLE"]
            selection_df = np.random.choice(a=activity_df[identifying_columns].drop_duplicates(),
                                            size=nr_agents, replace=False)
            activity_df = activity_df.merge(
                right=selection_df,
                on=identifying_columns,
                how="inner"
            )
            print('Dataframe reduced to', nr_agents, 'agents. Remaining activity_df size is', len(activity_df))
            activity_df = activity_df.reset_index(drop=True)

        # Set the travel mode for each row
        # -100 -> no travel
        activity_df["travel_mode"] = activity_df.apply(
            func=lambda x: set_travel_mode(
                x,
                travel_activities=self.__survey_encoding.travel_activities,
                unspecified_travel=self.__survey_encoding.unspecified_travel,
            ),
            axis=1,
        )

        activity_df["activity_description"] = activity_df.apply(
            func=lambda x: get_activity_description(
                x,
                activity_encoding=self.__survey_encoding.activity_encoding,
                travel_encoding=self.__survey_encoding.travel_encoding
            ),
            axis=1,
        )

        return activity_df

    def _aggregate_into_agenda(self, group: modin.DataFrame) -> tuple[Agenda, int]:
        """Aggregate activities of this group into an agenda."""
        # Sort by activity order
        group = group.sort_values(by="EPNUM", ascending=True)

        # Split the group's activities into windows and indicate the activity with the longest duration in each window
        # todo: if a travel activity does not have the longest duration in a segment, it is ignored, but should not be, workaround: short window length
        windows_dict = split_into_windows(
            starts=group["START"].to_numpy(),
            ends=group["END"].to_numpy(),
            window_length=self.__time_step_interval,
        )
        windows = np.asarray(tuple(windows_dict.values()), dtype=np.int16)
        windows_main_activity_ids = list(windows_dict.keys())

        # Create a list of the activities and overwrite travel activities with their travel_mode code
        windows_main_activities = group["MAIN"].iloc[windows_main_activity_ids].to_numpy(dtype=int)
        windows_travels = group["travel_mode"].iloc[windows_main_activity_ids].to_numpy(dtype=int)
        is_travel = windows_travels != -100
        windows_main_activities[is_travel] = windows_travels[is_travel]
        travel = np.argwhere(is_travel).ravel()

        persona_features = group[self.__persona_feature_names].drop_duplicates().iloc[0].to_dict()
        persid = group['PERSID'].drop_duplicates().iloc[0]
        starting_timestamp = self.get_valid_date(group)

        agenda = Agenda(
            activities=windows_main_activities,
            episode_ranges=windows,
            starting_timestamp=starting_timestamp,
            time_step_interval=self.__time_step_interval,
            travel=travel,
            persona_features=persona_features,
            persid=persid,
        )
        frequency = int(group["freq"].iloc[0])

        return agenda, frequency

    def get_valid_date(self, group: modin.DataFrame) -> datetime:
        """Extract year, month, and day or use dummy data."""
        year = group["YEAR"].iloc[0]
        month = (
            group["MONTH"].iloc[0]
            if group["MONTH"].notnull().iloc[0] and group["MONTH"].iloc[0] != -8
            else self.__dummy_date.month
        )
        # CDAY: calendar day ranging from 1 to 31, -8 (missing), or -9 (information is not available)
        # DAY: weekday ranging from 1 to 7 (Sun to Mon), -8 (missing), or -9 (unspecified weekday)
        day = (
            group["CDAY"].iloc[0]
            if group["CDAY"].notnull().iloc[0] and group["CDAY"].iloc[0] not in (-8, -9)
            else (
                self.__dummy_date.day
                if group["DAY"].iloc[0] in (-8, 9) or group["DAY"].isnull().iloc[0]
                else pick_random_cday(
                    year=year,
                    month=month,
                    weekday=((group["DAY"].iloc[0] - 2) % 7),
                    rng=self.__rng,
                )
            )
        )

        # todo: is this necessary since cday already verified above?
        # Try to create a valid datetime, correcting if needed.
        try:
            # todo: replace hardcoded hour=4 with clockst column from ipums
            starting_timestamp = datetime(year=year, month=month, day=day, hour=4)
        except ValueError:
            valid_timestamp = datetime(
                year=year,
                month=month,
                day=np.min((day, monthrange(year=year, month=month)[1])),
                hour=4
            )
            # get nearest past date with corresponding weekday
            if (
                    group.notna()["DAY"].iloc[0]
                    and group["DAY"].iloc[0] != valid_timestamp.weekday()
            ):
                offset = -(
                        (valid_timestamp.weekday() - ((group["DAY"].iloc[0] - 2) % 7)) % 7
                )
                starting_timestamp = valid_timestamp + timedelta(days=int(offset))
            else:
                starting_timestamp = valid_timestamp

        return starting_timestamp


def set_travel_mode(
        row: modin.Series,
        travel_activities: Iterable[Any],
        unspecified_travel: Iterable[Any],
) -> int:
    """
    Determine the travel mode for each activity, based on predefined travel activities. If MAIN activity value does 
    not correspond to a travel activity, travel mode is -100 (No travel). Else, if MTRAV travel code is valid 
    and indicates a travel activity as well, travel mode is inferred from MTRAV, else MAIN activity is used.

    Possible return values are:

        -100 No travel

        101 Travel by car etc
        102 Public transport
        103 Walk / on foot
        104 Other physical transport
        105 Other/unspecified transport

        11 Travel as a part of work
        43 Walking
        44 Cycling
        62 No activity, imputed or recorded transport
        63 Travel to/from work
        64 Education travel
        65 Voluntary/civic/religious travel
        66 Child/adult care travel
        67 Shop, person/hhld care travel
        68 Other travel
    """
    travel_mode = -100
    if row["MAIN"] in travel_activities:
        # MAIN indicates travel, but MTRAV indicates invalid, unspecific, or no travel
        if row["MTRAV"] < 0 or row["MTRAV"] + 100 in unspecified_travel:
            # use MAIN as travel mode, e.g., 'Walking' when 'Not travelling', or 'Ecuation travel' when 'Other/unspecified transport'
            travel_mode = row["MAIN"]
        else:
            # use MTRAV, because it indicates the transport mode, e.g., 'Travel by car etc', or 'Public transport'
            travel_mode = row["MTRAV"] + 100

    return travel_mode

def get_activity_description(
        row: modin.Series,
        activity_encoding: dict[int, str],
        travel_encoding: dict[int, str],
) -> str:
    activity_code = int(row["MAIN"])
    activity_description = ""
    if activity_code in activity_encoding:
        activity_description = activity_encoding[activity_code]
    elif activity_code in travel_encoding:
        activity_description = travel_encoding[activity_code]
    return activity_description