import numpy as np
from numpy import ndarray
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


def gravity_score(origin: ndarray | None, origin_relevance: float, destination: ndarray, destination_relevance: float
                  ) -> float:
    if origin is None:
        distance = 1
    else:
        distance = np.linalg.norm(origin - destination) / 1_000

    # set distance between identical origin and destination to not be zero
    small_non_zero_number = 1e-10
    if distance == 0:
        distance += small_non_zero_number

    score = origin_relevance * destination_relevance / np.power(distance, 2)
    return score


def calculate_od_gravity_scores(relevances: np.ndarray, locations: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise gravity scores p_ij between locations. p_ij indicates a probability of travelling between
    location i and j and is computed as:
        p_ij = r_i * r_j / d(i,j)**2
    where
        - r_i(j) is the relevance of location i(j), e.g., specified by a cells population count,
        - d(i,j) is the geographic distance between location i and j.

    Gravity scores are not normalized.

    See:
    Pappalardo, Luca, und Filippo Simini. „Data-Driven Generation of Spatio-Temporal Routines in Human Mobility“.
    Data Mining and Knowledge Discovery 32, Nr. 3 (Mai 2018): 787–829. https://doi.org/10.1007/s10618-017-0548-4.
    """

    # calculate a cell's relevance as the ratio of its population to the overall population in the tesselation (as
    # proposed by Pappalardo & Simini, p. 15)
    relevance = relevances / relevances.sum()

    relevance_cell_i = relevance.reshape(-1, 1)
    relevance_cell_j = relevance.reshape(1, -1)

    # calculate pairwise Euclidean distance between centroids
    distances = cdist(locations, locations, 'euclidean')
    squared_distances = np.power(distances, 2)

    # set distances to not be zero
    small_non_zero_number = 1e-10
    zero_values = np.argwhere(squared_distances == 0)
    for i, j in zero_values:
        squared_distances[i, j] += small_non_zero_number

    # compute gravity scores for each od pair
    gravity_scores = relevance_cell_i * relevance_cell_j / squared_distances

    # set gravity score to zero for identical cells
    for i, j in zero_values:
        gravity_scores[i, j] = 0

    return gravity_scores

def calculate_normalized_weighted_scores(scores: list, weights: list) -> np.ndarray:
    # make sure no score sums up to 0
    shifted_scores = [score + 1 if np.sum(score) == 0 else score for score in scores]

    # normalize
    normalized_scores = [score/np.sum(score) for score in shifted_scores]

    # stack scores into a single array
    normalized_scores_per_poi = np.column_stack(normalized_scores)

    # calculate weighted scores per poi
    weighted_scores_per_poi = np.power(normalized_scores_per_poi, weights)

    # calculate final score per poi as product of the scores per poi
    weighted_score_per_poi = np.prod(weighted_scores_per_poi, axis=1)

    # normalize
    normalized_sum_weighted_score_per_poi = weighted_score_per_poi / np.sum(weighted_score_per_poi)

    return normalized_sum_weighted_score_per_poi


def calculate_frequency_ranks(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate a ranking based on the frequency of each element in elements in descending order. A low score
    indicates a high frequency.
    """
    unique_elements, element_occurrence_ids, counts = np.unique(elements, return_index=True, return_counts=True, axis=0)
    # Calculate a ranking of the elements from the highest count to the lowest count
    ranks = rankdata(-counts, method='ordinal')
    return unique_elements, ranks


def recency_rank(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a ranking based on the recency of each element in elements from most recent to least recent."""
    # reverse array
    reversed_elements = elements[::-1]
    # Calculate the unique elements and the index of their first occurrence
    unique_elements, element_occurrence_ids = np.unique(reversed_elements, return_index=True, axis=0)
    element_sorting_id = range(len(unique_elements))
    unique_elements = [(unique_elements[i],
                        element_sorting_id[i],
                        element_occurrence_ids[i])
                       for i in range(len(unique_elements))]
    elements_sorted_by_occurrence = sorted(unique_elements, key=lambda x: x[2])

    # Add recency rank with a low rank indicating the most recent element, and sort by element sorting id
    ranks = np.arange(1, len(unique_elements) + 1)#, 0, -1)
    unique_elements_with_rank = sorted(zip(elements_sorted_by_occurrence, ranks), key=lambda x: x[0][1])

    unique_elements = np.asarray([el[0][0] for el in unique_elements_with_rank])
    ranks = np.asarray([el[1] for el in unique_elements_with_rank])
    return unique_elements, ranks
