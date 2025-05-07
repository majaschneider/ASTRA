import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from copy import copy
from datetime import datetime, timedelta
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Iterator, NamedTuple, Iterable


class AgendaSegment(NamedTuple):
    """
    An agenda segment describes a sequence of activities. Agenda segments are separated by travel activities.
    """

    activity_durations: dict[int, int]  # encoded activities, weighted by duration
    segment_range: tuple[int, int]  # define range of segment within an agenda
    travel_mode: int | None  # travel mode leading up to the segment
    travel_time_mins: int | None  # duration of previous travel

    @classmethod
    def from_agenda_elements(
        cls,
        activities: Iterable[int],
        episode_ranges: np.ndarray,
        travel_mode: int | None,
        travel_steps: int | None,
        scanning_interval: int,
    ) -> "AgendaSegment":
        """
        create an agenda segment from agenda elements
        """
        weights = np.abs(
            np.diff(episode_ranges, axis=1)
        ).ravel()  # determine durations of activities
        activity_durations = defaultdict(int)
        for element, weight in zip(activities, weights):  # assign weights to elements
            activity_durations[element] += weight
        segment_range = (
            int(episode_ranges[0, 0]),
            int(episode_ranges[-1, -1]),
        )  # determine total segment range

        return cls(
            activity_durations=activity_durations,
            segment_range=segment_range,
            travel_mode=travel_mode,
            travel_time_mins=(
                None if travel_steps is None else travel_steps * scanning_interval
            ),
        )


@dataclass(slots=True)
class Agenda:
    """
    stores a sequence of activities and associated information
    """

    # defines activities (each activity is a group of consecutive episodes of the same activity)
    activities: np.ndarray

    # defines activity episode ranges: (An activity episode refers to a specific period during which a particular
    # activity is continuously performed without interruption.)
    # index ranges of the agenda's activities
    episode_ranges: np.ndarray

    # dict of persona feature descriptions (e.g. AGE, SEX, ...) and integer encoded persona feature values
    persona_features: dict[str, int] | None

    # starting timestamp of agenda
    starting_timestamp: datetime

    # window_length of one time step in minutes
    time_step_interval: int

    # indices where activities are travel related or travel modes
    travel: np.ndarray

    persid: int

    def __len__(self) -> int:
        """
        return the number of activities in the agenda (this is not the number of unique activities)
        """
        return len(self.activities)

    @property
    def total_duration(self) -> timedelta:
        """
        calculate the total duration of the agenda in minutes
        """
        # the total duration is the product of the time step window_length size and the number of all activity episodes (
        # index of first non-included time step)
        return timedelta(
            minutes=self.time_step_interval * (self.episode_ranges[-1, 1] - 1)
        )

    @property
    def number_of_segments(self) -> int:
        """Get the number of agenda segments. Agenda segments consist of consecutive non-travel activities."""
        return len(self.travel) + 1

    def get_agenda_segment(self, segment_index: int) -> AgendaSegment:
        """Get the agenda segment at the given index."""
        assert 0 <= segment_index < self.number_of_segments

        # first segment of agenda
        if segment_index < 1:
            segments_first_activity_idx = 0

            # no travel activity done before this segment
            travel_mode = None
            travel_steps = None
        else:
            # get travel done before segment
            travel_idx = int(self.travel[segment_index - 1])

            # first activity index of segment is the first non travel index of segment
            segments_first_activity_idx = travel_idx + 1

            # get travel information
            travel_mode = self.activities[travel_idx]
            travel_steps = self.get_episode_len(int(travel_idx))

        # no travel activity done after the agenda
        last_idx = int(
            self.travel[segment_index]
            if segment_index < len(self.travel)
            else len(self)
        )

        agenda_segment = AgendaSegment.from_agenda_elements(
            activities=self.activities[segments_first_activity_idx:last_idx],
            episode_ranges=self.episode_ranges[segments_first_activity_idx:last_idx, :],
            travel_mode=travel_mode,
            travel_steps=travel_steps,
            scanning_interval=self.time_step_interval,
        )

        return agenda_segment

    def get_episode_len(self, idx: int) -> int:
        """Get length of episode at idx."""
        return int(np.ceil(self.episode_ranges[idx][1] - self.episode_ranges[idx][0]))

    def to_df(self) -> pd.DataFrame:
        """Convert agenda to time series dataframe."""
        targets = self.activities.copy()

        targets = np.fromiter(
            itertools.chain.from_iterable(
                (
                    (target for _ in range(self.get_episode_len(i)))
                    for i, target in enumerate(targets)
                )
            ),
            dtype=int,
        )

        timestamps = pd.date_range(
            start=self.starting_timestamp,
            freq="%sT" % self.time_step_interval,
            periods=self.episode_ranges[-1][1],
        )

        df = pd.DataFrame(data={"target": targets}, index=timestamps)

        return df

    def remove_consecutive_travel(self):
        try:
            while len(self.travel) > 0:
                # get consecutive indices of travel activities
                i = next(_get_consecutive_numbers(sequence=self.travel))

                # get corresponding activity ids
                i_activity = int(self.travel[i])
                j_activity = int(self.travel[i + 1])

                # get corresponding travel modes
                mode_i = int(self.activities[i_activity])
                mode_j = int(self.activities[j_activity])

                # use the mode whose episode is longer
                episode_lengths = np.diff(self.episode_ranges[[i_activity, j_activity]], axis=1)
                mode = mode_i if episode_lengths[0] > episode_lengths[1] else mode_j

                self.merge_consecutive_episodes(i=i_activity, activity=mode, is_travel=True)

        except StopIteration:
            ...

    def merge_consecutive_episodes(self, i: int, activity: int, is_travel: bool):
        new_episode_ranges = self.episode_ranges.copy()
        # set new bounds for first episode
        new_episode_start = self.episode_ranges[i, 0]
        new_episode_end = self.episode_ranges[i + 1, 1]
        new_episode_ranges[i] = np.asarray([new_episode_start, new_episode_end])
        # remove second episode
        new_episode_ranges = np.delete(new_episode_ranges, i + 1, axis=0)
        self.episode_ranges = new_episode_ranges

        new_activities = self.activities.copy()
        # replace first activity and delete second activity
        new_activities[i] = activity
        new_activities = np.delete(self.activities, i + 1)
        self.activities = new_activities

        if is_travel:
            new_travel = self.travel[(self.travel <= i) | (self.travel > i + 1)]
            new_travel[new_travel > i + 1] -= 1
            self.travel = new_travel

    def preprocess_travel(self):
        if len(self.travel) > 0:
            self.remove_consecutive_travel()

        # remove initial travel activity
        if len(self.travel) > 0:
            if self.travel[0] == 0:
                self.activities = self.activities[1:]
                self.episode_ranges = self.episode_ranges[1:] - self.episode_ranges[1, 0]
                self.travel = self.travel[1:] - 1

        # remove final travel activity
        if len(self.travel) > 0:
            if self.travel[-1] == len(self) - 1:
                self.activities = self.activities[:-1]
                self.episode_ranges = self.episode_ranges[:-1, :]
                self.travel = self.travel[:-1]


def _get_consecutive_numbers(sequence: Sequence[int]) -> Iterator[int]:
    """Return an iterator over the index of the first number in each consecutive number pair in sequence."""
    ids = [i for i, (a, b) in enumerate(itertools.pairwise(sequence)) if b - a == 1]
    for i in ids:
        yield i
