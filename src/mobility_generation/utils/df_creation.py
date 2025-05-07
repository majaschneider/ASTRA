import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import timedelta
from skmob import TrajDataFrame
from shapely import Point

from src.mobility_generation.data_classes.agent import Agent
from src.survey_processing.dataclasses.agenda import AgendaSegment
from src.survey_processing.loading.travel_survey_encoding import SurveyEncoding
from src.tessellation.data_classes.pois import Poi


def to_agenda_segment_df(agent: Agent, agenda_segments: list[AgendaSegment], survey_encoding: SurveyEncoding) -> pd.DataFrame:

    route_df = []
    for segment in agenda_segments:
        activity_ids = np.asarray(list(segment.activity_durations.keys()))
        activity_durations = np.asarray(list(segment.activity_durations.values()))
        duration_minutes = int(np.sum(activity_durations))
        activity_descriptions = list(survey_encoding.decode_activities(activities=list(activity_ids)))

        activities = []
        for activity_id in activity_ids:
            activity = survey_encoding.encode_activity(activity=activity_id)
            travel = survey_encoding.encode_travel_mode(travel_mode=activity_id)
            if activity is not None:
                activities.append(activity)
            elif travel is not None:
                activities.append(travel)

        data = {'duration_minutes': duration_minutes, 'activity_ids': [activity_ids.tolist()],
                'activities': [activities], 'activity_durations': [activity_durations.tolist()],
                'activity_descriptions': [activity_descriptions],
                'persid': str(agent.agenda.persid)}

        segment_df = pd.DataFrame(data=data)
        route_df.append(segment_df)

    df = pd.concat(route_df)
    df = df.reset_index(drop=True)

    return df

def to_df(agent: Agent, agent_id: int, poi_route: list[Poi], survey_encoding: SurveyEncoding) -> pd.DataFrame:
    agenda = agent.agenda

    route_df = []
    for i in range(agent.agenda.number_of_segments):
        episodes_starts = agenda.episode_ranges[:, 0]
        segment_start = agent.history.segment_indices[i][0]
        segment_end = agent.history.segment_indices[i][1] - 1
        poi = poi_route[i]
        datetime_min = agenda.starting_timestamp + timedelta(minutes=segment_start * agenda.time_step_interval)
        datetime_max = agenda.starting_timestamp + timedelta(minutes=segment_end * agenda.time_step_interval)

        # correct timestamps for start and end of day because surveys were taken for time range from 4am to 4am next day
        if i == 0:
            datetime_min = datetime.datetime(datetime_min.year, datetime_min.month, datetime_min.day)
        if i == agent.agenda.number_of_segments - 1:
            datetime_max = datetime.datetime(datetime_min.year, datetime_min.month, datetime_min.day, 23, 55)
        diff = datetime_max - datetime_min
        duration_minutes = divmod(diff.days * 24 * 60 * 60 + diff.seconds, 60)[0]

        segment_is_a_return = i in agent.history.returns.keys()
        return_to_segment_id = agent.history.returns[i] if segment_is_a_return else -1
        activity_idxs = list(np.nonzero((segment_start <= episodes_starts) & (episodes_starts <= segment_end))[0])
        activity_ids = np.asarray([agenda.activities[idx] for idx in activity_idxs])
        activity_descriptions = list(survey_encoding.decode_activities(activities=list(activity_ids)))

        activities = []
        for activity_id in activity_ids:
            activity = survey_encoding.encode_activity(activity=activity_id)
            travel = survey_encoding.encode_travel_mode(travel_mode=activity_id)
            if activity is not None:
                activities.append(activity)
            elif travel is not None:
                activities.append(travel)

        travel_mode = list(agent.history.travel_modes.values())[i - 1] if i > 0 else None

        data = {'agent_id': agent_id,
                'lng': poi.longitude, 'lat': poi.latitude,
                'datetime_min': datetime_min, 'datetime_max': datetime_max, 'duration_minutes': duration_minutes,
                'poi_description': poi.poi_sentence, 'poi_name': poi.poi_name,
                'segment_id': i, 'is_return': segment_is_a_return, 'return_to_segment_id': return_to_segment_id,
                'activity_ids': [list(set(activity_ids.tolist()))], 'activities': [list(set(activities))],
                'activity_descriptions': [list(set(activity_descriptions))],
                'travel_mode': travel_mode}

        poi_df = pd.DataFrame(data=data)
        route_df.append(poi_df)

    df = pd.concat(route_df)
    df = df.reset_index(drop=True)

    return df

def segment_data_to_df(
    segment_range: tuple[int, int],
    segment_id: int,
    travel_mode: str,
    agent: Agent,
    poi: Poi,
    survey_encoding: SurveyEncoding,
) -> pd.DataFrame:

    agenda = agent.agenda
    route = agent.history
    episodes_starts = agenda.episode_ranges[:, 0]
    segment_start = segment_range[0]
    segment_end = segment_range[1] - 1

    poi_df = {"longitude": poi.longitude, "latitude": poi.latitude, "poi": poi.poi_sentence, "poi_name": poi.poi_name}

    activity_idxs_in_this_segment = np.nonzero((segment_start <= episodes_starts) & (episodes_starts <= segment_end))[0]
    # for each activity
    data = []
    for i in activity_idxs_in_this_segment:
        activity_poi = poi_df.copy()
        activity_poi["segment_id"] = segment_id
        activity_poi["travel_mode"] = travel_mode    # travel_mode to reach this segment
        segment_is_a_return = segment_id in route.returns.keys()
        activity_poi["is_return"] = segment_is_a_return
        activity_poi["return_to_segment_id"] = route.returns[segment_id] if segment_is_a_return else -1
        activity_poi["activity_id"] = int(agenda.activities[i])
        activity_poi["timestamp"] = pd.date_range(
            start=agenda.starting_timestamp +
                  timedelta(minutes=agenda.time_step_interval * int(agenda.episode_ranges[i, 0])),
            periods=agenda.get_episode_len(i),
            freq="%sT" % agenda.time_step_interval
        )
        data.append(activity_poi)

    # infer data for this segment from all activities
    df = pd.DataFrame(data=data)
    activities = df["activity_id"].to_list()
    df["activity"] = list(survey_encoding.decode_activities(activities=activities))
    df["activity_description"] = list(survey_encoding.encode_activities(activities=activities))
    df["travel_description"] = list(survey_encoding.encode_travel_modes(travel_modes=activities))
    # convert to dataframe with user-defined time interval per row
    df = df.explode(column="timestamp", ignore_index=True)

    return df


def to_tdf(agent: Agent, poi_route: list[Poi], survey_encoding: SurveyEncoding) -> TrajDataFrame:
    travel_modes = [None] + list(agent.history.travel_modes.values())

    route_df = []
    for i in range(agent.agenda.number_of_segments):
        segment_df = segment_data_to_df(
                segment_range=agent.history.segment_indices[i],
                segment_id=i,
                travel_mode=travel_modes[i],
                survey_encoding=survey_encoding,
                poi=poi_route[i],
                agent=agent,
            )
        route_df.append(segment_df)

    df = pd.concat(route_df)
    tdf = TrajDataFrame(data=df, latitude="latitude", longitude="longitude", datetime="timestamp").reset_index()

    return tdf

def group_tdf(tdf: TrajDataFrame) -> gpd.GeoDataFrame:
    gdf = pd.DataFrame(tdf).groupby(['lat', 'lng', 'segment_id', 'poi', 'poi_name', 'is_return',
                                     'return_to_segment_id', 'travel_mode']).agg(
        min_value=('datetime', 'min'),
        max_value=('datetime', 'max'),
        activities=('activity_description', set),
        activity_ids=('activity_id', set)
    ).sort_values(by='segment_id').reset_index()

    gdf['duration'] = gdf['max_value'] - gdf['min_value']
    gdf['duration'] = gdf.apply(lambda x: int(timedelta.total_seconds(x['duration'])/60.), axis=1)
    gdf['datetime'] = gdf['min_value']

    gdf['geometry'] = gdf.apply(lambda x: Point(x['lng'], x['lat']), axis=1)
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry').reset_index()
    gdf.activities = gdf.activities.apply(list)
    gdf.activity_ids = gdf.activity_ids.apply(list)

    return gdf
