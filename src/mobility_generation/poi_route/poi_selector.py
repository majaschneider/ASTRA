from typing import NamedTuple

import numpy as np
import ray

from src.tessellation.data_classes.pois import Poi
from src.tessellation.tessellation.tessellation import Tessellation
from src.mobility_generation.utils.scoring import gravity_score


class PoiSelector(NamedTuple):
    """Provides functionality to select a POI in a cell of the tesselation."""

    rng: np.random.Generator

    def select_osm_poi(
        self,
        embedding_id: int,
        tessellation: Tessellation,
        poi_candidates: np.ndarray[Poi],
        previous_cell_idx: int | None,
    ) -> Poi:

        poi_sentences = [poi.poi_sentence for poi in poi_candidates]

        # get embedding ids of available osm pois
        poi_embedding_ids = ray.get(
            tessellation.embedding_collection.get_ids.remote(
                elements=poi_sentences
            )
        )

        # filter poi candidates with matching poi sentence
        idxs_of_matching_poi_embedding_ids = np.where((poi_embedding_ids.ravel() - embedding_id) == 0)[0]
        poi_candidates = poi_candidates[idxs_of_matching_poi_embedding_ids]
        if len(idxs_of_matching_poi_embedding_ids) == 0:
            print('no POIs with matching id', embedding_id, 'in available POIs:', poi_embedding_ids)

        if previous_cell_idx is None:
            origin = None
            origin_relevance = 1
        else:
            previous_poi_set = tessellation.get_poi_set(previous_cell_idx)
            # todo: origin needs to be prev POI, not cell centroid
            origin = np.asarray([previous_poi_set.centroid_lon, previous_poi_set.centroid_lat])
            # todo: relevance of prev POI iso 1
            origin_relevance = 1

        # calculate score for every candidate
        poi_candidate_scores = [
            np.asarray(
                gravity_score(
                    origin=origin,
                    origin_relevance=origin_relevance,
                    destination=np.asarray([poi.longitude, poi.latitude]),
                    destination_relevance=float(poi.relevance_score)
                )
            ).reshape(1, -1)
            for poi in poi_candidates
        ]

        # select candidate with a probability proportional to the score
        poi_candidate_scores = np.concatenate(poi_candidate_scores, axis=0)

        # normalize
        poi_candidate_scores = poi_candidate_scores / poi_candidate_scores.sum(axis=0)

        if poi_candidate_scores.ndim > 1:
            poi_candidate_scores = poi_candidate_scores.prod(axis=1)
            poi_candidate_scores = poi_candidate_scores / poi_candidate_scores.sum()

        chosen_idx = self.rng.choice(len(poi_candidate_scores), p=poi_candidate_scores)
        poi = poi_candidates[chosen_idx]

        return poi
