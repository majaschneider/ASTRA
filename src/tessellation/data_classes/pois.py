from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Poi:
    longitude: float
    latitude: float
    poi_sentence: str
    relevance_score: float
    poi_name: str
