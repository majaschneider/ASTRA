import itertools
import tomli
import warnings
from pathlib import Path

import numpy as np
from typing import Any, Hashable, NamedTuple, Iterable, Iterator

import xxhash


class SurveyEncoding(NamedTuple):
    """
    class defining encoding of travel survey data
    """

    activity_encoding: dict[int, str]  # encoding of travel survey activities
    activity_decoding: dict[int, str]  # decoding of travel survey activities

    travel_encoding: dict[int, str]  # encoding of travel survey travel modes
    travel_decoding: dict[int, str]  # decoding of travel survey travel modes

    travel_activities: tuple[int | Any, ...]
    unspecified_travel: tuple[int | Any, ...]
    endpoints: tuple[int | Any, ...]

    seed: int

    @classmethod
    def from_config(cls,
                    encoding_path: Path,
                    seed: int) -> "SurveyEncoding":
        """
        read survey encoding from config file
        """
        with encoding_path.open("rb") as f:
            encoding_data = tomli.load(f)

        activity_encoding = {
            int(k): v for k, v in encoding_data["encoding"]["activity"].items()
        }

        if len(activity_encoding) == 0:
            warnings.warn(
                message="No activity encoding was provided, leaving survey activities unencoded",
                category=UserWarning,
            )

        travel_encoding = {
            int(k): v for k, v in encoding_data["encoding"]["travel_modes"].items()
        }

        if len(travel_encoding) == 0:
            warnings.warn(
                message="No travel encoding was provided, leaving travel activities unencoded",
                category=UserWarning,
            )

        activity_decoding = {
            int(k): v for k, v in encoding_data["decoding"]["activity"].items()
        }
        travel_decoding = {
            int(k): v for k, v in encoding_data["decoding"]["travel_modes"].items()
        }

        assert len(activity_decoding) and len(travel_decoding)

        travel_activities = tuple(
            (
                int(k) if str(k).isnumeric() else k
                for k in itertools.chain.from_iterable(
                    encoding_data["properties"]["travel_activities"].items()
                )
            )
        )

        unspecified_travel = tuple(
            (
                int(k) if str(k).isnumeric() else k
                for k in itertools.chain.from_iterable(
                    encoding_data["properties"]["unspecified_travel"].items()
                )
            )
        )
        endpoints = tuple(
            (
                int(k) if str(k).isnumeric() else k
                for k in itertools.chain.from_iterable(
                    encoding_data["properties"]["endpoints"].items()
                )
            )
        )
        return cls(
            activity_encoding=activity_encoding,
            activity_decoding=activity_decoding,
            travel_encoding=travel_encoding,
            travel_decoding=travel_decoding,
            unspecified_travel=unspecified_travel,
            endpoints=endpoints,
            seed=seed,
            travel_activities=travel_activities,
        )

    def encode_activities(self, activities: Iterable[int]) -> Iterator[str | Any]:
        for activity in activities:
            yield self.encode_activity(activity)

    def encode_activity(self, activity: int) -> str | Any:
        return self.activity_encoding.get(activity, None)

    def encode_travel_modes(self, travel_modes: Iterable[int]) -> Iterator[str | Any]:
        for travel_mode in travel_modes:
            yield self.encode_travel_mode(travel_mode)

    def encode_travel_mode(self, travel_mode: int) -> str | Any:
        return self.travel_encoding.get(travel_mode, None)

    def encode_persona_features(self, persona_features: dict[str, Any]) -> dict[str, Any]:
        return {k: self.encode_persona_feature(v) for k, v in persona_features.items()}

    def encode_persona_feature(self, persona_feature: Hashable) -> int | float:
        if isinstance(persona_feature, int | float):
            return persona_feature
        else:
            hasher = xxhash.xxh3_64(seed=self.seed)
            hasher.update(str(persona_feature))
            return hasher.intdigest()

    def decode_activities(self, activities: Iterable[int]) -> Iterator[str]:
        for activity in activities:
            yield self.decode_activity(activity)

    def decode_activity(self, encoded_activity: int) -> str:
        return self.activity_decoding.get(encoded_activity)

    def decode_travel_mode(self, encoded_travel_mode: int) -> str:
        return self.travel_decoding.get(encoded_travel_mode)

    def get_valid_agenda_element(self, value: float) -> int:
        value = round(value)
        if value in self.activity_decoding.keys() or value in self.travel_decoding.keys():
            return value
        else:
            warnings.warn(message=f"Unknown agenda element '{value}'")
            return 70

    def all_encoded(self) -> np.ndarray:
        return np.asarray(list(self.activity_decoding.keys()) + list(self.travel_decoding.keys()))
