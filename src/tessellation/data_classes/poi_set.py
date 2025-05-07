from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class POISet:
    """Dense representation of a collection of POIs."""

    # centroid of all POIs in this collection
    centroid_lon: float
    centroid_lat: float

    # unique sentences describing the POIs in this collection
    poi_sentences: tuple[str, ...]

    # frequency of each POI sentence
    poi_sentence_frequencies: np.ndarray

    # mean distance of all POIs to the collections centroid
    mean_distance_to_centroid: float

    # embedding ids for unique pois
    embedding_ids: np.ndarray

    def get_number_of_pois(self):
        return len(self.poi_sentences)
