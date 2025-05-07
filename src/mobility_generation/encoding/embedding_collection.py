from typing import Iterable
from pathlib import Path

import csv
import numpy as np
import ray
import torch
from faiss import (
    write_index,
    read_index,
    IndexFlatIP,
)
from sentence_transformers import SentenceTransformer

from src.survey_processing.loading.travel_survey_encoding import SurveyEncoding


@ray.remote
class EmbeddingCollection:
    """
    class for creating and storing sentence embeddings.
    Iterative actor when used across multiple processes.
    """

    # survey encoding for decoding encoded survey activities
    __survey_encoding: SurveyEncoding

    # sentence transformer: creates embeddings from sentences (word count limit: see documentation of pretrained index)
    __sentence_transformer: SentenceTransformer

    # index for storing sentence embeddings
    __embedding_index: IndexFlatIP

    # stores embedding strs for alongside their embedding index ids
    __str_id_mapping: dict[str, int]

    # embeddings of the travel activities drive, bicycle and walk
    __travel_embeddings: np.ndarray

    # save paths for memory mapping and loading previous data
    __embedding_index_save_path: Path | None
    __str_id_mapping_save_path: Path | None

    # random number generator for probabilistic selection of travel mode
    __rng: np.random.Generator

    # if set the weighted average embedding is created for an activity segment.
    # Otherwise, the activity with the longest duration is embedded
    __use_weighted_average_search: bool
    # if this option is set each weight is squared for weighted average embedding
    __weight_impact_on_average: bool

    # dimension of embedding vector
    embedding_dimension: int

    def __init__(
        self,
        survey_encoding: SurveyEncoding,
        rng: np.random.Generator,
        use_weighted_average_search: bool,
        weight_impact_on_average: bool,
        index_path: Path,
        model_name: str = "multi-qa-MiniLM-L6-cos-v1",
        embedding_dimension: int = 384,
    ) -> None:
        self.__survey_encoding = survey_encoding
        self.__rng = rng
        self.__use_weighted_average_search = use_weighted_average_search
        self.__weight_impact_on_average = weight_impact_on_average
        self.embedding_dimension = embedding_dimension

        # load pretrained transformer of given name
        self.__sentence_transformer = SentenceTransformer(model_name)

        # create flat faiss index for inner product search
        self.__embedding_index = IndexFlatIP(self.embedding_dimension)

        self.__str_id_mapping = dict()

        # create embeddings corresponding to the modes drive, bicycle and walk
        self.__travel_embeddings = self.__sentence_transformer.encode(
            sentences=[
                "taking the car or public transport",
                "taking the bicycle",
                "walking",
            ],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # create folder structure for memory mapping and saving the embeddings index and the str to id mapping
        index_path.mkdir(exist_ok=True)
        self.__embedding_index_save_path = index_path / "embeddings.index"
        self.__str_id_mapping_save_path = index_path / "str_id_mapping.csv"

        # load embedding index if both index and mapping exist
        if self.__embedding_index_save_path.exists() and self.__str_id_mapping_save_path.exists():
            # read embedding index
            self.__embedding_index = read_index(str(self.__embedding_index_save_path))

            # read mapping from str elements to ids
            with self.__str_id_mapping_save_path.open("r") as f:
                str_id_mapping = csv.reader(f)
                str_id_mapping = {
                    mapping[0]: int(mapping[1])
                    for mapping in str_id_mapping if len(mapping) == 2
                }
            self.__str_id_mapping = str_id_mapping
        else:
            # create new index
            write_index(self.__embedding_index, str(self.__embedding_index_save_path))

    def get_ids(self, elements: Iterable[str]) -> np.ndarray:
        """Get or embed if not exists."""
        return np.fromiter(
            (
                self._get_id(element=element, create_missing=True)
                for element in elements
            ),
            dtype=int,
        )

    def _get_id(self, element: str, create_missing: bool) -> int | None:
        """
        get embedding index id for a str element
        if create_missing is set, a new embedding is created and saved to the index in case of unknown str element
        """
        # get id from mapping
        index_id = self.__str_id_mapping.get(element, None)

        # if element_str is unknown and create_missing is set, a new embedding is created, saved and its id returned
        if index_id is None and create_missing:
            # create embedding with sentence transformer (normalize for inner product search)
            embedding = self.__sentence_transformer.encode(
                sentences=element.lower(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            # id of the unknown element will be the current index size
            index_id = int(self.__embedding_index.ntotal)
            # add str element and id to str to id mapping
            self.__str_id_mapping[element] = index_id
            # add embedding to index
            self.__embedding_index.add(embedding.reshape((1, -1)))
        elif create_missing:
            index_id = int(index_id)

        return index_id

    def _get_embedding(self, sentence: str) -> np.ndarray:
        """Get or create embedding of this sentence."""
        # get or create an embedding in the index
        index_id = self._get_id(sentence, create_missing=True)
        embedding = self.__embedding_index.reconstruct(index_id)

        return embedding

    def get_travel_mode(self, encoded_travel_mode: int, prefix: str) -> str:
        """
        Make sure, that the travel mode can be decoded to be either 'drive', 'bicycle' or 'walk'. If travel mode can
        not be properly decoded to a supported mode, check whether it is a travel activity code and calculate
        the similarity of the decoded value with each mode and choose travel mode according to its likelihood. If the
        travel mode is no travel activity either, choose the default travel mode 'drive'.
        """
        supported_travel_modes = ('drive', 'bicycle', 'walk')
        default_travel_mode = supported_travel_modes[0]

        decoded_travel_mode = self.__survey_encoding.decode_travel_mode(encoded_travel_mode)

        # if travel mode is not one of the supported modes
        if decoded_travel_mode not in supported_travel_modes:
            # if travel mode is a travel activity
            if encoded_travel_mode in self.__survey_encoding.travel_activities:
                decoded_travel_activity = self.__survey_encoding.decode_activity(encoded_activity=encoded_travel_mode)

                # choose a supported mode based on its similarity to the agent's persona features and travel activity
                embedded_travel_activity = self._get_embedding(sentence=prefix + decoded_travel_activity)
                similarities = np.dot(self.__travel_embeddings, embedded_travel_activity.ravel()).ravel()
                mode_idx = self.__rng.choice(a=len(similarities), p=similarities / similarities.sum())
                decoded_travel_mode = supported_travel_modes[mode_idx]
            # otherwise, choose default travel mode
            else:
                decoded_travel_mode = default_travel_mode

        return decoded_travel_mode

    def _get_weighted_average_embedding(
        self,
        sentences: Iterable[str],
        weights: np.ndarray,
        prefix: str,
    ) -> np.ndarray:
        """Calculate weighted average embedding of sentences."""
        embeddings = np.concatenate(
            [self._get_embedding(prefix + sentence).reshape(-1, 1) for sentence in sentences],
            axis=1,
        )

        weighted_average_embedding = np.average(embeddings, axis=1, weights=weights)

        return weighted_average_embedding

    def find_most_similar_pois(
        self,
        encoded_activities: np.ndarray,
        weights: np.ndarray,
        prefix: str,
        ids_to_search: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search a faiss index of target queries for a search query and return their similarity and index. A search query
        consists of a string containing a prefix (e.g., persona features) and activities.
        If __use_weighted_average_search is True, all given activities are used for the search query and weighted with
        the given weights (e.g., their duration). If __use_weighted_average_search is False, the activity with the
        highest weight is used only as a search query. Target queries contain POI descriptions. The search query is
        compared only to target queries, whose activity code is listed in ids_to_search.
        """
        decoded_activities = list(self.__survey_encoding.decode_activities(encoded_activities))

        if self.__use_weighted_average_search:
            # normalize
            weights = weights / weights.sum()

            # raise to higher power and normalize
            weights = np.power(weights, self.__weight_impact_on_average)
            weights = weights / weights.sum()

            embedded_query_sentence = self._get_weighted_average_embedding(
                sentences=decoded_activities,
                weights=weights,
                prefix=prefix
            )
        else:
            # get activity with the highest weight
            activity, _ = max(zip(decoded_activities, weights), key=lambda x: x[1])
            embedded_query_sentence = self._get_embedding(sentence=prefix + activity)

        similarities, indices = self._search_embedding_index(
            embedded_query_sentence=embedded_query_sentence,
            ids_to_search=ids_to_search
        )

        return similarities, indices

    def _search_embedding_index(
        self,
        embedded_query_sentence: np.ndarray,
        ids_to_search: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the embedded query sentence in the index' embeddings whose embedding id is in ids_to_search. Return
        the k most similar embeddings and return similarity scores and the embedding ids.
        """
        similarities, indices = self.__embedding_index.search(
            embedded_query_sentence.reshape(1, -1),
            k=self.__embedding_index.ntotal
        )
        # TODO: makeshift solution since IDSelectorArray does not work as intended
        similarities = similarities.ravel()
        indices = indices.ravel()
        mask = np.isin(indices, ids_to_search)
        return similarities[mask], indices[mask]

    def save(self) -> None:
        """
        save embedding index and str to id mapping to disk
        """

        # write faiss index
        write_index(self.__embedding_index, str(self.__embedding_index_save_path))

        # save str to id mapping as csv
        with self.__str_id_mapping_save_path.open("w") as f:
            writer = csv.writer(f)
            for element, idx in self.__str_id_mapping.items():
                writer.writerow([element, idx])
        return
