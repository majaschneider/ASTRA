import pandas as pd
from shapely.lib import normalize
from skmob.measures import individual, collective
from skmob import TrajDataFrame
import numpy as np


class PlotDataFrame(pd.DataFrame):
    is_aggregated: bool = False
    x_name: str
    y_name: str
    x_label: str
    y_label: str
    title: str
    xscale: str
    xticks: np.ndarray

    def __init__(self, *args, x_name: str, y_name: str, x_label: str, y_label: str, y_label_hist: str, title: str,
                 is_aggregated: bool= False, xscale: str='linear', xticks: np.ndarray=None,  **kwargs):
        super().__init__(*args, **kwargs)
        self.is_aggregated = is_aggregated
        self.x_name = x_name
        self.y_name = y_name
        self.x_label = x_label
        self.y_label = y_label
        self.y_label_hist = y_label_hist
        self.title = title
        self.xscale = xscale
        self.xticks = xticks


class Metrics:

    def cumulative_trip_distance(self, df: pd.DataFrame) -> PlotDataFrame:
        """Computes the cumulative geographical travel distance in kilometers for every individual in tdf."""
        df = individual.distance_straight_line(df, show_progress=False)
        df.sort_values(by='distance_straight_line', ascending=False, inplace=True)
        df.rename(columns={'distance_straight_line': 'cumulative_trip_distance'}, inplace=True)
        df = PlotDataFrame(df, x_name='uid', y_name='cumulative_trip_distance', x_label='User id',
                           y_label='Cumulative trip distance [km]', y_label_hist='P(#Users)',
                           title='Cumulative trip distance', is_aggregated=False, xscale='log')
        return df

    def trip_distance(self, df: pd.DataFrame) -> PlotDataFrame:
        """
        Computes the geographical travel distance between consecutive POIs in a user's trajectory in kilometers for
        every individual in tdf.
        """
        df = individual.jump_lengths(df, show_progress=False)
        df = df.explode('jump_lengths')
        df = df[df['jump_lengths'].isna() == False]
        df.sort_values(by='jump_lengths', ascending=False, inplace=True)
        df.rename(columns={'jump_lengths': 'trip_distance'}, inplace=True)
        df = PlotDataFrame(df, x_name='uid', y_name='trip_distance', x_label='User id',
                           y_label='Trip distance [km]', y_label_hist='P(#Users)', title='Trip distance',
                           is_aggregated=False, xscale='log')
        return df

    def radius_of_gyration(self, df: pd.DataFrame) -> PlotDataFrame:
        """The radius of gyration is the characteristic distance traveled by an individual during the period of
        observation (Gonzalez et al., 2008; Pappalardo et al., 2013b, 2015b). In detail, it characterizes the spatial
        spread of the locations visited by an individual from the trajectories’ center of mass.
        """
        df = individual.radius_of_gyration(df, show_progress=False)
        df.sort_values(by='radius_of_gyration', ascending=False, inplace=True)
        df = PlotDataFrame(df, x_name='uid', y_name='radius_of_gyration', x_label='User id', 
                           y_label='Radius of gyration [km]', y_label_hist='P(#Users)', title='Radius of gyration', 
                           is_aggregated=False, xscale='log')
        return df

    def mobility_entropy(self, df: pd.DataFrame) -> PlotDataFrame:
        """
        The mobility entropy of an individual is defined as the Shannon entropy of a user's visited locations
        (Song et al., 2010b; Eagle and Pentland, 2009; Pappalardo et al., 2016b).
        """
        df = individual.uncorrelated_entropy(df, show_progress=False, normalize=True)
        df.sort_values(by='norm_uncorrelated_entropy', ascending=True, inplace=True)
        df.rename(columns={'norm_uncorrelated_entropy': 'mobility_entropy'}, inplace=True)
        df = PlotDataFrame(df, x_name='uid', y_name='mobility_entropy', x_label='User id',
                           y_label='Mobility entropy', y_label_hist='P(#Users)', title='Mobility entropy',
                           is_aggregated=False)
        return df

    def location_frequency(self, df: pd.DataFrame) -> PlotDataFrame:
        """
        The probability of a user u visiting a location l_i in their trajectory (the user’s visitation frequency of l_i).
        It is defined as
            p_i = n_i/|L(u)|
        where
            n_i is the number of visits to l_i, and
            L(u) is the set of locations visited by user u.
        """
        location_frequency = individual.location_frequency(df, show_progress=False, as_ranks=True)
        df = pd.DataFrame(data={'rank': range(len(location_frequency)), 'location_frequency': location_frequency})
        # df.sort_values(by='location_frequency', ascending=False, inplace=True)
        # # multiple users can visit the same location but their frequencies should be counted individually
        # df['location_id'] = df.groupby(['lat', 'lng']).ngroup()
        df = PlotDataFrame(df, x_name='rank', y_name='location_frequency', x_label='Rank',
                           y_label='Location frequency', y_label_hist='', title='Location frequency',
                           is_aggregated=True, xscale='log')
        return df

    def visits_per_location(self, df: pd.DataFrame) -> PlotDataFrame:
        """The number of visits by all users (also repeated visits) per location."""
        df = collective.visits_per_location(df)
        df.rename(columns={'n_visits': 'visits_per_location'}, inplace=True)
        df.sort_values(by='visits_per_location', ascending=True, inplace=True)
        df['location_id'] = df.groupby(['lat', 'lng']).ngroup()
        df = df[['location_id', 'visits_per_location']]
        df = PlotDataFrame(df, x_name='location_id', y_name='visits_per_location', x_label='Location id',
                           y_label='Number of visiting users', y_label_hist='P(#Locations)', title='Visits per location',
                           is_aggregated=False, xscale='log')
        return df

    def locations_per_user(self, df: pd.DataFrame) -> PlotDataFrame:
        """
        The number of locations (data points) visited by an individual during the period of observation describes the
        degree of exploration of an individual.
        """
        df = individual.number_of_visits(df, show_progress=False)
        df.sort_values(by='number_of_visits', ascending=False, inplace=True)
        df.rename(columns={'number_of_visits': 'locations_per_user'}, inplace=True)
        df = PlotDataFrame(df, x_name='uid', y_name='locations_per_user', x_label='User id',
                           y_label='Number of locations', y_label_hist='P(#Users)', title='Locations per user',
                           is_aggregated=False)
        return df

    def trips_per_hour(self, df: pd.DataFrame) -> PlotDataFrame:
        """
        The number of trips (data points) per hour of all individuals. Human movements follow the circadian rhythm, i.e.,
        they are prevalently stationary during the night and move preferably at specific times of the day
        (Gonźalez et al., 2008; Pappalardo et al., 2013b).
        """
        df = collective.visits_per_time_unit(df)
        df['datetime'] = df.index
        df['time'] = df['datetime'].dt.time
        df_g = df.groupby('time').agg(trips_per_hour=('n_visits', 'sum'))
        times = pd.date_range("00:00", "23:00", freq="1h").time
        trips_per_hour_dict = df_g.trips_per_hour.to_dict()
        y = [trips_per_hour_dict[t] if t in trips_per_hour_dict.keys() else 0 for t in times]
        x = [t.strftime("%H") for t in times]
        df = pd.DataFrame(data=zip(x, y), columns=['hour', 'trips_per_hour'])
        df = PlotDataFrame(df, x_name='hour', y_name='trips_per_hour', x_label='Hour', y_label='P(#Trips)',
                           y_label_hist='', title='Trips per hour', is_aggregated=True,
                           xticks = np.arange(0, 24, 5))
        return df

    def trips_per_day(self, df: pd.DataFrame) -> PlotDataFrame:
        """The number of trips (data points) per calendar day of all individuals."""
        df = collective.visits_per_time_unit(df, time_unit='d')
        df['datetime'] = df.index
        df['day'] = df.datetime.dt.day
        df_g = df.groupby('day').agg(trips_per_day=('n_visits', 'sum'))
        x = [wd for wd in range(31)]
        trips_per_day_dict = df_g.trips_per_day.to_dict()
        y = [trips_per_day_dict[t] if t in trips_per_day_dict.keys() else 0 for t in x]
        df = pd.DataFrame(data=zip(x, y), columns=['day', 'trips_per_day'])
        df = PlotDataFrame(df, x_name='day', y_name='trips_per_day', x_label='Calendar day',
                           y_label='P(#Trips)', y_label_hist='', title='Trips per day', is_aggregated=True)
        return df

    def trips_per_weekday(self, df: pd.DataFrame) -> PlotDataFrame:
        """The number of trips (data points) per week day of all individuals. Monday = 0, Tuesday = 1, ..., Sunday = 6"""
        df = collective.visits_per_time_unit(df, time_unit='d')
        df['datetime'] = df.index
        df['weekday'] = df.datetime.dt.weekday
        df_g = df.groupby('weekday').agg(trips_per_day=('n_visits', 'sum'))
        x = [wd for wd in range(7)]
        trips_per_day_dict = df_g.trips_per_day.to_dict()
        y = [trips_per_day_dict[t] if t in trips_per_day_dict.keys() else 0 for t in x]
        df = pd.DataFrame(data=zip(x, y), columns=['weekday', 'trips_per_weekday'])
        df = PlotDataFrame(df, x_name='weekday', y_name='trips_per_weekday', x_label='Weekday',
                           y_label='P(#Trips)', y_label_hist='', title='Trips per weekday',
                           is_aggregated=True)
        return df

    def time_of_stays(self, df: pd.DataFrame) -> PlotDataFrame:
        """
        The distribution of stay times in hours. Stay time is the amount of time an individual spends at a
        particular location, measured as the stay duration between two consecutive data points.
        """
        waiting_times = individual.waiting_times(df, show_progress=False, merge=True)
        df = pd.DataFrame(data=waiting_times, columns=['time_of_stays'])
        df['time_of_stays'] = df['time_of_stays'] / 3600.
        df['stay_id'] = df.index
        df = PlotDataFrame(df, x_name='stay_id', y_name='time_of_stays', x_label='Stay (user and location)',
                           y_label='Duration [h]', y_label_hist='P(#Stays)', title='Stay duration', is_aggregated=False)
        return df

    def calculate(self, df: pd.DataFrame):
        """
        T Trips per hour
        D Trips per day
        ∆t Time of stays
        V Visits per location
        N Locations per user
        f (L) Location frequency
        ∆r Trip distance
        Cumulative trip distance
        rg Radius of gyration
        Sunc Mobility entropy
        """
        metrics = {
            'trips_per_hour': self.trips_per_hour(df),
            'trips_per_day': self.trips_per_day(df),
            'trips_per_weekday': self.trips_per_weekday(df),
            'time_of_stays': self.time_of_stays(df),
            "visits_per_location": self.visits_per_location(df),
            "locations_per_user": self.locations_per_user(df),
            "location_frequency": self.location_frequency(df),
            "trip_distance": self.trip_distance(df),
            "cumulative_trip_distance": self.cumulative_trip_distance(df),
            "radius_of_gyration": self.radius_of_gyration(df),
            "mobility_entropy": self.mobility_entropy(df),
        }
        return metrics
