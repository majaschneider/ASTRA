import numpy as np

from calendar import monthrange
from itertools import groupby
from datetime import datetime, timedelta
from math import ceil, floor


def split_into_windows(window_length: int, starts: np.ndarray, ends: np.ndarray) -> dict[int, tuple[int, int]]:
    """Split into windows of window_length."""
    # dict keys: id of activity with the longest duration in a window
    # dict value: window range (0,1)-> 0 minutes to window_length minutes
    assert len(starts) == len(ends)

    # determine overall bounds for all ranges
    start_minute = np.min(starts)
    end_minute = ceil(np.max(ends) / window_length) * window_length + 1

    # create discrete bounds for elements (cols: element_starts, element_ends)
    windows = np.concatenate(
        (
            np.arange(start=start_minute, stop=end_minute - window_length, step=window_length).reshape((-1, 1)),
            np.arange(start=start_minute + window_length, stop=end_minute, step=window_length).reshape((-1, 1))
         ), axis=1
    )
    starts = starts.reshape((-1, 1))
    ends = ends.reshape((-1, 1))
    nr_activities = len(starts)
    nr_windows = len(windows)

    # assign intervals to activities
    indices = []
    for i in range(nr_windows):
        interval_start, interval_end = windows[i, :]
        interval_ends = np.full((nr_activities, 1), fill_value=interval_end)
        interval_starts = np.full((nr_activities, 1), fill_value=interval_start)
        vectors = np.concatenate((interval_starts, interval_ends, starts, ends), axis=1)
        total_interval_lengths = (np.max(vectors, axis=1) - np.min(vectors, axis=1)).reshape((-1, 1))

        interval_coverage = (ends - starts + interval_ends - interval_starts) - total_interval_lengths
        indices.append(np.argmax(interval_coverage))

    result = {}
    current_index = 0
    for k, g in groupby(indices):
        new_index = current_index + len(list(g))
        result.update({int(k): (current_index, new_index)})
        current_index = new_index

    return result


def discretize_new(
        window_length: int, start_times: np.ndarray, end_times: np.ndarray
) -> dict[int, tuple[int, int]]:
    """
    Discretizes time intervals into fixed-size intervals and assigns the input intervals
    to the closest fitting discrete time range.

    The function maps the start and end times of a series of activities to predefined
    fixed-size intervals (e.g., 30-minute slots). Each activity will be associated with
    the window_length it overlaps with the most.

    Parameters:
    -----------
    window_length : int
        The fixed window_length length (in minutes) that will be used to discretize the time range.
    start_times : np.ndarray
        A 1D array containing the start times of activities.
    end_times : np.ndarray
        A 1D array containing the end times of activities. Must be the same length as `start_times`.

    Returns:
    --------
    dict[int, tuple[int, int]]:
        A dictionary where the key is the discrete window_length index and the value is a tuple
        indicating the range (start, end) of activities that fall within that window_length.
    """

    # Ensure that the start_times and end_times arrays have the same length
    assert len(start_times) == len(end_times), "Start and end times arrays must be of the same length."

    # Determine the overall bounds for the time range
    earliest_start = np.min(start_times)
    latest_end = ceil(np.max(end_times) / window_length) * window_length + 1  # Round up to nearest window_length

    # Create discrete window_length boundaries
    discrete_intervals = np.concatenate(
        (
            np.arange(start=earliest_start, stop=latest_end - window_length, step=window_length).reshape((-1, 1)),
            np.arange(start=earliest_start + window_length, stop=latest_end, step=window_length).reshape((-1, 1)),
        ),
        axis=1,
    )

    # Reshape start_times and end_times to align dimensions for processing
    start_times = start_times.reshape((-1, 1))
    end_times = end_times.reshape((-1, 1))

    interval_assignments = []

    # Iterate over the discrete intervals and find the best match for each activity
    for interval_idx in range(discrete_intervals.shape[0]):
        interval_start, interval_end = discrete_intervals[interval_idx, :]

        # Create arrays that match the shape of the start_times/end_times arrays for the current window_length
        interval_end_repeated = np.full((start_times.shape[0], 1), fill_value=interval_end)
        interval_start_repeated = np.full((start_times.shape[0], 1), fill_value=interval_start)

        # Stack the vectors and compute the total length covered by the intervals
        combined_times = np.concatenate((interval_start_repeated, interval_end_repeated, start_times, end_times),
                                        axis=1)
        total_interval_lengths = (np.max(combined_times, axis=1) - np.min(combined_times, axis=1)).reshape((-1, 1))

        # Compute how much of each activity overlaps with the current window_length
        overlap_length = (
                                     end_times - interval_start_repeated + interval_end_repeated - start_times) - total_interval_lengths

        # Append the index of the activity that overlaps most with this window_length
        interval_assignments.append(np.argmax(overlap_length))

    # Group activities by their most fitting window_length and create the result dictionary
    result = {}
    current_index = 0
    for interval_id, group in groupby(interval_assignments):
        group_length = len(list(group))
        new_index = current_index + group_length
        result[int(interval_id)] = (current_index, new_index)
        current_index = new_index

    return result


def pick_random_cday(weekday: int, month: int, year: int, rng: np.random.Generator) -> int:
    """Randomly choose a calendar day in the given month and year, that matches the given weekday."""
    # get the calendar day in the first week of the month that matches the requested weekday
    nr_cdays_in_week = 7
    last_date_of_first_week = datetime(year, month, nr_cdays_in_week)
    distance_to_given_weekday = float((last_date_of_first_week.weekday() - weekday) % nr_cdays_in_week)
    first_matching_date = last_date_of_first_week - timedelta(days=distance_to_given_weekday)
    first_matching_cday = first_matching_date.day

    # calculate available weeks, randomly choose one and get the matching day
    nr_cdays_in_month = monthrange(year=year, month=month)[1]
    available_weeks = floor((nr_cdays_in_month - first_matching_cday) / nr_cdays_in_week)
    random_week = rng.choice(available_weeks)
    random_cday = first_matching_cday + random_week * nr_cdays_in_week

    return random_cday
