from typing import NamedTuple

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field


class RouteElement(NamedTuple):
    cell_idx: int
    embedding_idx: int
    segment_range: tuple[int, int]
    idx_of_first_visit: int | None = None
    travel_mode: str | None = None


@dataclass(slots=True)
class Route:
    """
    A route captures information about the stops that have been visited by an agent. A stop refers to a segment in the
    agent's agenda, consisting of a number of activities. Stops are connected by travel activity and assigned to a
    specific cell in a tesselation. A route further stores information about when the agent returns to previously
    visited cells, which travel modes have been used, and about the semantic information of the stop (referenced by the
    index of the stop's embedded value in an embedding register).
    """

    # indices of visited cells in tesselation in visiting order
    cell_idxs: list[int] = field(default_factory=list)

    # indices of activities for each segment
    segment_indices: list[tuple[int, int]] = field(default_factory=list)

    # travel modes used for travel between the stops
    travel_modes: dict[int, str] = field(default_factory=dict)

    # segment_ids that indicate a return along with the segment_id that the return refers to
    returns: dict[int, int] = field(default_factory=dict)

    # chosen embedding id for each stop
    embedding_ids: list[int] = field(default_factory=list)

    def segment_ids_without_returns(self) -> list[int]:
        return [segment_id for segment_id, _ in enumerate(self.cell_idxs) if segment_id not in self.returns]

    def is_unvisited(self, ids: np.ndarray) -> np.ndarray:
        """
        Return those elements from the given array of embedding ids, that have not been visited so far in this route.
        """
        visited_locations = np.asarray(
            tuple(
                zip(
                    self.cell_idxs,
                    self.embedding_ids,
                )
            ),
        )
        return np.fromiter(
            (np.all(np.any(idx != visited_locations, axis=1))
             for idx in ids),
            dtype=bool,
        )

    @property
    def equals_last(self) -> np.ndarray:
        locations = np.asarray(
            tuple(
                zip(
                    self.cell_idxs,
                    self.embedding_ids,
                )
            ),
        )
        return np.all(locations == locations[-1, :], axis=1)

    @property
    def last_cell_idx(self) -> int:
        last = None
        if len(self.cell_idxs):
            last = self.cell_idxs[-1]
        return last

    @property
    def last_embedding(self) -> int:
        last = None
        if len(self.cell_idxs):
            last = self.embedding_ids[-1]
        return last

    @property
    def num_unique_locations(self) -> int:
        """
        get number of unique stop locations
        """
        return len(self.cell_idxs) - len(self.returns)

    def append_element(self, element: RouteElement) -> None:
        """
        append stop to route
        """
        self.cell_idxs.append(int(element.cell_idx))
        self.embedding_ids.append(int(element.embedding_idx))
        self.segment_indices.append(element.segment_range)
        if element.idx_of_first_visit is not None:
            self.returns[len(self.cell_idxs) - 1] = int(element.idx_of_first_visit)
        if element.travel_mode is not None:
            self.travel_modes[len(self.cell_idxs) - 1] = element.travel_mode
        return

    def __len__(self) -> int:
        """
        length of route
        """
        return len(self.cell_idxs)
