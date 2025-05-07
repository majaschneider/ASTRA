from xmlrpc.client import Error
import os
from typing import Any

import psycopg2
from geopandas import GeoDataFrame
import pandas as pd
from skmob import TrajDataFrame
from sqlalchemy import create_engine
from psycopg2 import pool, Binary
from psycopg2.extras import execute_values
import pickle
from shapely import Polygon
import networkx as nx
import csv
from bs4 import BeautifulSoup

from src.evaluation.metrics import PlotDataFrame
from src.mobility_generation.data_classes.agent import Agent


class Database:

    engine = None

    host = None
    port = None
    database = None
    user = None
    password = None

    def __init__(self, host, port, database, user, password):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection_string = f"dbname={database} user={user} password={password} host={host} port={port}"

        self.init_engine()

    def get_new_engine(self):
        engine = create_engine(
            url='postgresql+psycopg2://' + self.user + ':' + self.password + '@' + self.host + ':' + self.port + '/' +
                self.database,
            pool_size=2,
            max_overflow=2,
            pool_pre_ping=True
        )
        return engine

    def init_engine(self):
        if self.engine is None:
            self.engine = self.get_new_engine()

    def execute_with_new_connection(self, sql, returning=False, data=None) -> Any | None:
        conn = self.create_connection()
        try:
            with (conn.cursor() as cursor):
                if data is None:
                    cursor.execute(sql)
                else:
                    execute_values(cursor, sql, data)
                if returning:
                    if cursor.rowcount > 0:
                        return cursor.fetchone()[0]
                    else:
                        raise Exception("No data returned")
                else:
                    return None
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error while connecting to PostgreSQL:", error)
            if conn:
                conn.rollback()
            else:
                print("Couldn't rollback.")
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.cursor().close()
                conn.close()
            else:
                print("Connection already closed.")

    def create_connection(self):
        conn = psycopg2.connect(
            user=self.user,
            password=self.password,
            host=self.host,
            dbname=self.database,
            port=self.port
        )
        conn.autocommit = True
        return conn

    def create_cells_table(self):
        sql = """CREATE TABLE IF NOT EXISTS cells (cell_id SERIAL PRIMARY KEY, area_str VARCHAR(500), 
                area BYTEA, UNIQUE (area));"""
        self.execute_with_new_connection(sql)

    def append_cell(self, area: Polygon):
        area_bytea = Binary(pickle.dumps(area))
        sql = f"""INSERT INTO cells(area_str, area) 
                VALUES ('{str(area)}', {area_bytea}) 
                RETURNING cell_id;"""
        cell_id = self.execute_with_new_connection(sql, returning=True)
        return cell_id

    def get_cell_area(self, cell_id: int) -> Polygon | None:
        sql = f"""SELECT area FROM cells WHERE cell_id = '{str(cell_id)}';"""
        df = pd.read_sql(sql, self.engine)
        if df.empty:
            area = None
        else:
            area_bytea = df['area'].iloc[0]
            area = pickle.loads(area_bytea)
        return area

    def get_cell_id(self, area: Polygon) -> int | None:
        sql = f"""SELECT cell_id FROM cells WHERE area_str = '{str(area)}';"""
        df = pd.read_sql(sql, self.engine)
        if df.empty:
            cell_id = None
        else:
            cell_id = df['cell_id'].iloc[0]
        return cell_id

    def delete_cell(self, area: Polygon) -> None:
        sql = f"""DELETE FROM cells WHERE area_str = '{str(area)}';"""
        self.execute_with_new_connection(sql)

    def create_osm_data_table(self):
        sql = """CREATE TABLE IF NOT EXISTS osm_data (cell_id INTEGER, osm_gdf BYTEA, is_empty BOOLEAN, 
                CONSTRAINT fk_cell_id FOREIGN KEY (cell_id) REFERENCES cells(cell_id));"""
        self.execute_with_new_connection(sql)

    def append_osm_data(self, cell_id: int, osm_gdf: GeoDataFrame | None, is_empty: bool):
        """Save the area and the gdf in the database. If gdf is None or empty, an empty GeoDataFrame will be saved."""
        if osm_gdf is None:
            osm_gdf = GeoDataFrame()
        serialized_gdf = Binary(pickle.dumps(osm_gdf))
        sql = f"""INSERT INTO osm_data(cell_id, osm_gdf, is_empty) 
                VALUES ('{str(cell_id)}', {serialized_gdf}, {is_empty});"""
        self.execute_with_new_connection(sql)

    def get_osm_data(self, cell_id: int) -> GeoDataFrame | None:
        """
        Return the area's corresponding GeoDataFrame (can be empty if the saved GeoDataFrame was None or empty). If
        no value for the area was found, return None.
        """
        sql = f"""SELECT osm_gdf FROM osm_data WHERE cell_id = '{str(cell_id)}';"""
        df = pd.read_sql(sql, self.engine)
        if df.empty:
            gdf = None
        else:
            serialized_gdf = df['osm_gdf'].iloc[0]
            gdf = pickle.loads(serialized_gdf)
        return gdf

    def create_travel_times_table(self, use_euclidean_travel=False):
        addition = ''
        if use_euclidean_travel:
            addition = 'euclidean_'
        sql = f"""CREATE TABLE IF NOT EXISTS {addition}travel_times (transport_mode VARCHAR(20), 
                origin_osm_id VARCHAR(20), destination_osm_id VARCHAR(20), travel_time REAL,
                UNIQUE (transport_mode, origin_osm_id, destination_osm_id));"""
        self.execute_with_new_connection(sql)

    def append_travel_times(self, transport_mode: str, travel_times: dict[(int, int): float], use_euclidean_travel=False):
        """Save the travel time from origin to destination using the respective transport mode in the database."""
        addition = ''
        if use_euclidean_travel:
            addition = 'euclidean_'
        data = [(transport_mode, str(key[0]), str(key[1]), travel_times.get(key)) for key in travel_times.keys()]
        sql = f"""INSERT INTO {addition}travel_times(transport_mode, origin_osm_id, destination_osm_id, travel_time) 
                VALUES %s ON CONFLICT (transport_mode, origin_osm_id, destination_osm_id) DO NOTHING;"""
        self.execute_with_new_connection(sql, data)

    def get_travel_times(self, node_pairs: list, transport_mode: str, use_euclidean_travel=False) -> dict[(int, int): float]:
        """
        Return the travel times for each origin-destination pair for the respective area and transport mode. If no
        value was found, return None.
        """
        addition = ''
        if use_euclidean_travel:
            addition = 'euclidean_'
        batches = list(chunk_data(data=node_pairs, batch_size=1000))
        df_result = pd.DataFrame()
        for batch in batches:
            node_pairs_str = str([(str(a), str(b)) for (a, b) in batch]).replace('[', '(').replace(']', ')')
            sql = f"""SELECT origin_osm_id, destination_osm_id, travel_time 
                    FROM {addition}travel_times 
                    WHERE (origin_osm_id, destination_osm_id) IN {node_pairs_str} 
                    AND transport_mode = '{str(transport_mode)}';"""
            df = pd.read_sql(sql, self.engine)
            df_result = pd.concat([df_result, df])
        result = { (int(row.origin_osm_id), int(row.destination_osm_id)): row.travel_time for row in df_result.itertuples(index=False) }
        return result


    def create_nearest_nodes_table(self):
        sql = """CREATE TABLE IF NOT EXISTS nearest_nodes (cell_id INTEGER, transport_mode VARCHAR(20),
                osm_id VARCHAR(20), 
                CONSTRAINT fk_cell_id FOREIGN KEY (cell_id) REFERENCES cells(cell_id),
                UNIQUE (cell_id, transport_mode, osm_id));"""
        self.execute_with_new_connection(sql)

    def append_nearest_nodes(self, transport_mode: str, cell_to_nearest_nodes: dict[int: int]):
        """Save the cell id and its nearest node."""
        data = [(str(key), transport_mode, str(cell_to_nearest_nodes.get(key))) for key in cell_to_nearest_nodes.keys()]
        sql = f"""INSERT INTO nearest_nodes(cell_id, transport_mode, osm_id) 
                VALUES %s ON CONFLICT (cell_id, transport_mode, osm_id) DO NOTHING;"""
        self.execute_with_new_connection(sql, data)

    def get_nearest_nodes(self, grid_id: int, transport_mode: str) -> dict[int: int]:
        """
        Return cell ids and their nearest osm node for the respective area and transport mode. If no value was found,
        return None.
        """
        sql = f"""SELECT a.cell_id, a.osm_id 
                FROM nearest_nodes a left join cell_grid_relation b on a.cell_id = b.cell_id 
                WHERE b.grid_id = '{grid_id}' AND transport_mode = '{str(transport_mode)}';"""
        df = pd.read_sql(sql, self.engine)
        cell_to_node = { int(row.cell_id): int(row.osm_id) for row in df.itertuples(index=False) }
        return cell_to_node

    def create_osmnx_graphs_table(self):
        sql = """CREATE TABLE IF NOT EXISTS osmnx_graphs (grid_id INTEGER, transport_mode VARCHAR(20), graph oid, 
            CONSTRAINT fk_grid_id FOREIGN KEY (grid_id) REFERENCES grids(grid_id), 
            UNIQUE (grid_id, transport_mode));"""
        self.execute_with_new_connection(sql)

    def drop_osmnx_graphs_table(self):
        """
        Drop osmnx_graphs table. Makes sure, that contained large objects are deleted before the table is deleted to
        avoid orphaned large objects.
        """
        sql = """DELETE FROM osmnx_graphs;"""
        self.execute_with_new_connection(sql)
        sql = """DROP TABLE IF EXISTS osmnx_graphs;"""
        self.execute_with_new_connection(sql)

    def append_osmnx_graph(self, grid_id: int, transport_mode: str, graph: nx.MultiDiGraph):
        """Save the OSMnx graph as a bytea object."""
        graph_bytea = Binary(pickle.dumps(graph))
        sql = f"""INSERT INTO osmnx_graphs(grid_id, transport_mode, graph) 
                VALUES ('{grid_id}', '{transport_mode}', lo_from_bytea(0, {graph_bytea})) 
                ON CONFLICT (grid_id, transport_mode) DO NOTHING;"""
        self.execute_with_new_connection(sql)

    def get_osmnx_graph(self, grid_id: int, transport_mode: str) -> nx.MultiDiGraph | None:
        """
        Return the OSMnx graph stored as a large object for the respective area and transport mode. If no data found,
        return None.
        """
        sql = f"""SELECT lo_get(graph) as graph FROM osmnx_graphs 
                WHERE grid_id = '{grid_id}' AND transport_mode = '{str(transport_mode)}';"""
        df = pd.read_sql(sql, self.engine)
        if df.empty:
            return None
        graph_bytea = df.iloc[0]['graph']
        graph = pickle.loads(graph_bytea)
        return graph

    def create_grid_table(self) -> None:
        sql = """CREATE TABLE IF NOT EXISTS grids (grid_id SERIAL PRIMARY KEY, area VARCHAR(2000), cell_size INTEGER,
                UNIQUE (area, cell_size));"""
        self.execute_with_new_connection(sql)

    def append_grid(self, area: Polygon, cell_size: float) -> int | None:
        """Save the area and its id in the database."""
        sql = f"""INSERT INTO grids(area, cell_size) VALUES ('{area}', {cell_size}) 
                ON CONFLICT (area, cell_size) DO NOTHING RETURNING grid_id;"""
        grid_id = self.execute_with_new_connection(sql, returning=True)
        return grid_id

    def get_grid_id(self, area: Polygon, cell_size: float) -> int | None:
        """Return the area id of area. If no value was found, return None."""
        sql = f"""SELECT grid_id FROM grids WHERE area = '{str(area)}' AND cell_size = {cell_size};"""
        df = pd.read_sql(sql, self.engine)
        if df.empty:
            grid_id = None
        else:
            grid_id = int(df.grid_id)
        return grid_id

    def create_cell_grid_relation_table(self) -> None:
        sql = """CREATE TABLE IF NOT EXISTS cell_grid_relation (grid_id INTEGER, cell_id INTEGER, 
                CONSTRAINT fk_grid_id FOREIGN KEY (grid_id) REFERENCES grids(grid_id),
                CONSTRAINT fk_cell_id FOREIGN KEY (cell_id) REFERENCES cells(cell_id),
                UNIQUE (grid_id, cell_id));"""
        self.execute_with_new_connection(sql)

    def append_cell_grid_relation(self, grid_id: int, cell_id: int) -> None:
        sql = f"""INSERT INTO cell_grid_relation(grid_id, cell_id) 
                VALUES ('{grid_id}', '{cell_id}') 
                ON CONFLICT (grid_id, cell_id) DO NOTHING;"""
        self.execute_with_new_connection(sql)

    def create_astra_routes_checkin_table(self) -> None:
        sql = """CREATE TABLE IF NOT EXISTS astra_routes_checkin (agent_id INTEGER, 
                lng DECIMAL, lat DECIMAL, datetime_min TIMESTAMP, datetime_max TIMESTAMP, duration_minutes INTEGER, 
                poi_description VARCHAR(2000), poi_name VARCHAR(500), 
                segment_id INTEGER, is_return BOOL, return_to_segment_id INTEGER, 
                activity_ids INTEGER[], activities VARCHAR(500)[], activity_descriptions VARCHAR(500)[], 
                travel_mode VARCHAR(100), 
                CONSTRAINT fk_agent_id FOREIGN KEY (agent_id) REFERENCES astra_agents(agent_id));"""
        self.execute_with_new_connection(sql)

    def append_astra_routes_checkin(self, df: pd.DataFrame) -> None:
        """Append astra route data on checkin level."""
        df.to_sql('astra_routes_checkin', self.engine, if_exists='append', index=False)

    def get_astra_routes_checkin(self, config_id: int) -> TrajDataFrame:
        sql = f"""SELECT a.* FROM astra_routes_checkin a 
                INNER JOIN astra_agents b ON a.agent_id = b.agent_id 
                WHERE b.config_id = '{config_id}';"""
        df = pd.read_sql(sql, self.engine)
        if df.empty:
            tdf = None
        else:
            # df['datetime_avg'] = df.datetime_min + (df.datetime_max - df.datetime_min) / 2
            tdf = TrajDataFrame(data=df, latitude='lat', longitude='lng', datetime="datetime_min", user_id="agent_id")
        return tdf

    def create_astra_agents_table(self) -> None:
        sql = """CREATE TABLE IF NOT EXISTS astra_agents (agent_id SERIAL PRIMARY KEY, persid text, config_id INTEGER, 
                agent BYTEA, 
                CONSTRAINT fk_persid FOREIGN KEY (persid) REFERENCES demographics_mtus(persid));"""
        self.execute_with_new_connection(sql)

    def append_astra_agents(self, persid: int, config_id: int, agent: Agent) -> int | None:
        agent_bytea = Binary(pickle.dumps(agent))
        sql = f"""INSERT INTO astra_agents(persid, config_id, agent) 
                VALUES ('{persid}', '{config_id}', {agent_bytea}) 
                RETURNING agent_id;"""
        agent_id = self.execute_with_new_connection(sql, returning=True)
        return agent_id

    def get_astra_agent_id(self, persid: int, config_id: int, agent: Agent) -> int:
        agent_bytea = Binary(pickle.dumps(agent))
        sql = f"""SELECT agent_id FROM astra_agents WHERE persid = '{persid}' AND config_id = '{config_id}' AND 
                agent = {agent_bytea};"""
        df = pd.read_sql(sql, self.engine)
        if df.empty:
            agent_id = None
        else:
            agent_id = df['agent_id']
        return agent_id

    def create_evaluation_table(self) -> None:
        sql = """CREATE TABLE IF NOT EXISTS evaluation (config_id INTEGER, dataset_name VARCHAR(100), 
                metric_name VARCHAR(100), metric_df BYTEA);"""
        self.execute_with_new_connection(sql)

    def append_evaluation(self, config_id: int, dataset_name: str, metric_name: str, metric_df: pd.DataFrame) -> None:
        metric_df_bytea = Binary(pickle.dumps(metric_df))
        sql = f"""INSERT INTO evaluation(config_id, dataset_name, metric_name, metric_df) 
                VALUES ('{config_id}', '{dataset_name}', '{metric_name}', {metric_df_bytea});"""
        self.execute_with_new_connection(sql)

    def drop_evaluation_table(self) -> None:
        sql = """DROP TABLE evaluation;"""
        self.execute_with_new_connection(sql)

    def get_config_ids(self) -> list[int]:
        sql = f"""SELECT DISTINCT config_id FROM evaluation;"""
        df = pd.read_sql(sql, self.engine)
        return list(df['config_id'])

    def get_evaluation(self, config_id: int) -> dict[str, [str, PlotDataFrame]]:
        sql = f"""SELECT dataset_name, metric_name, metric_df 
                FROM evaluation WHERE config_id = '{config_id}';"""
        df = pd.read_sql(sql, self.engine)
        eval_dict = None
        if not df.empty:
            eval_dict = {}
            for dataset_name in df.dataset_name.unique():
                eval_dict[dataset_name] = {}
                df_filter = df.copy()
                df_filter = df_filter[df_filter['dataset_name'] == dataset_name]
                for metric_name in df_filter.metric_name.unique():
                    metric_df_bytea = df_filter[df_filter['metric_name'] == metric_name]['metric_df'].iloc[0]
                    eval_dict[dataset_name][metric_name] = pickle.loads(metric_df_bytea)
        return eval_dict

    def create_evaluation_values_table(self) -> None:
        sql = """CREATE TABLE IF NOT EXISTS evaluation_values (config_id INTEGER, dataset_name VARCHAR(100), 
                metric_name VARCHAR(100), uid INTEGER, metric_value DECIMAL);"""
        self.execute_with_new_connection(sql)

    def append_evaluation_values(self, config_id: int, dataset_name: str, metric_name: str, metric_df: pd.DataFrame) -> None:
        metric_df['metric_value'] = metric_df[metric_name]
        metric_df['metric_name'] = metric_name
        metric_df['dataset_name'] = dataset_name
        metric_df['config_id'] = config_id
        metric_df = metric_df[['config_id', 'dataset_name', 'metric_name', 'metric_value']]
        metric_df.to_sql('evaluation_values', self.engine, if_exists='append', index=False)

    def get_astra_agents(self) -> dict[int, Agent]:
        """Return the agent data for the given grid id and config id. If no value was found, return None."""
        sql = f"""SELECT agent_id, agent FROM astra_agents;"""
        df = pd.read_sql(sql, self.engine)
        if df.empty:
            agents_dict = None
        else:
            agents = [pickle.loads(agent) for agent in df.agent]
            agents_dict = {agent_id: agent for (agent_id, agent) in zip(list(df['agent_id']), agents)}
        return agents_dict

    def create_ditras_routes_table(self) -> None:
        sql = """CREATE TABLE IF NOT EXISTS ditras_routes (uid INTEGER, datetime TIMESTAMP, lat DECIMAL, lng DECIMAL);"""
        self.execute_with_new_connection(sql)

    def append_ditras_routes(self, dfg: TrajDataFrame) -> None:
        """Append ditras route data. dfg should contain columns uid, datetime, lat, lng."""
        dfg.to_sql('ditras_routes', self.engine, if_exists='append', index=False)

    def get_ditras_routes(self) -> TrajDataFrame:
        """Return the agent data. If no value was found, return None."""
        sql = f"""SELECT uid, datetime, lat, lng FROM ditras_routes;"""
        df = pd.read_sql(sql, self.engine)
        if df.empty:
            dfg = None
        else:
            dfg = TrajDataFrame(data=df, latitude='lat', longitude='lng', datetime="datetime", user_id="uid")
        return dfg

    def create_config_grid_table(self) -> None:
        sql = """CREATE TABLE IF NOT EXISTS config_grid (config_id INTEGER, grid_id INTEGER, 
                UNIQUE (config_id, grid_id));"""
        self.execute_with_new_connection(sql)

    def append_config_grid(self, config_id: int, grid_id: int) -> None:
        sql = f"""INSERT INTO config_grid(config_id, grid_id) 
                VALUES ('{config_id}', '{grid_id}') 
                ON CONFLICT (config_id, grid_id) DO NOTHING;"""
        self.execute_with_new_connection(sql)

    def create_agenda_segments_table(self) -> None:
        sql = """CREATE TABLE IF NOT EXISTS agenda_segments (agenda_segment_id SERIAL PRIMARY KEY, 
                duration_minutes INTEGER, activity_ids INTEGER[], activities VARCHAR(500)[], 
                activity_durations INTEGER[], activity_descriptions VARCHAR(500)[], persid text);"""
        self.execute_with_new_connection(sql)

    def append_agenda_segments(self, route_df: pd.DataFrame) -> None:
        """Save the agenda segment and the agent's persid in the database."""
        route_df.to_sql('agenda_segments', self.engine, if_exists='append', index=False)

    def get_agenda_segments_with_demographics(self, activity_ids: list[list[int]]|None = None) -> pd.DataFrame | None:
        """Return all agenda segments."""
        if activity_ids is not None:
            activity_ids = str([str(ids) for ids in activity_ids]).replace('[', '{').replace(']', '}')
            activity_ids = '(' + activity_ids[1:-1] + ')'
        sql = f"""SELECT a.activity_ids, a.activities, a.activity_durations, a.activity_descriptions, 
                b."AGE", b."SEX", b."CITIZEN", b."EMPSTAT", b."STUDENT", b."INCOME", b."CIVSTAT", b."EDTRY", b."CARER", 
                b."NCHILD", b."URBAN", b."SINGPAR", b."AGEKID", b."DISAB", b."ALONE", b."CHILD", b."SPPART", b."OAD"                  
                FROM agenda_segments a INNER JOIN (
                        SELECT DISTINCT "PERSID", "AGE", "SEX", "CITIZEN", "EMPSTAT", "STUDENT", "INCOME", "CIVSTAT", 
                        "EDTRY", "CARER", "NCHILD", "URBAN", "SINGPAR", "AGEKID", "DISAB", "ALONE", "CHILD", "SPPART", 
                        "OAD"
                        FROM activities
                    ) AS b ON a.persid = cast(b."PERSID" as text) 
                {'WHERE a.activity_ids IN ' + activity_ids if activity_ids is not None else ''}
                ;"""
        df = pd.read_sql(sql, self.engine)
        if df.empty:
            df = None
        return df

    def get_most_frequent_activities(self) -> pd.DataFrame | None:
        sql = """SELECT activity, COUNT(*)
                FROM agenda_segments, unnest(activities) AS activity
                GROUP BY activity
                ORDER BY COUNT(*) DESC;"""
        df = pd.read_sql(sql, self.engine)
        if df.empty:
            df = None
        return df

    def get_most_frequent_agenda_segments(self) -> pd.DataFrame | None:
        sql = """SELECT sorted_activity_ids, COUNT(*)
                FROM (
                    SELECT ARRAY(SELECT unnest(activity_ids) ORDER BY 1) AS sorted_activity_ids
                    FROM agenda_segments
                ) AS subquery
                GROUP BY sorted_activity_ids
                ORDER BY COUNT(*) DESC;"""
        df = pd.read_sql(sql, self.engine)
        if df.empty:
            df = None
        return df

    def read_citi_bike(self, df: pd.DataFrame) -> None:
        df.to_sql('citi_bike', con=self.engine, index=False, if_exists='append')

    def read_activities(self, df: pd.DataFrame) -> None:
        df.to_sql('activities', con=self.engine, index=False)

    def read_demographics_into_table(self) -> None:
        file = 'resources/mtus/mtus_00007.csv'
        sql = '''CREATE TABLE demographics_mtus(SAMPLE text, IDENT text, COUNTRY text, HLDID text, PERSID text PRIMARY KEY, 
        DIARY text, YEAR int, PARNTID1 text, PARNTID2 text, PARTID text, RELREFP text, DAY int, MONTH int, HHTYPE text, 
        HHLDSIZE text, NCHILD text, FAMSTAT text, SINGPAR text, OWNHOME text, URBAN text, AGEKID text, AGEKID2 text, 
        AGE int, SEX text, CITIZEN text, CIVSTAT text, COHAB text, EDTRY text, EDUCA text, CARER text, EMPSTAT text, 
        EMPSP text, SECTOR text, EMP text, UNEMP text, WORKHRS text, RETIRED text, STUDENT text, DISAB text, 
        INCOME text, INCORIG text, OCCUPO text, OCOMBWT text, PROPWT text, DIARYTYPE text, BADCASE text, 
        ACT_CHCARE text, ACT_CIVIC text, ACT_EDUCA text, ACT_INHOME text, ACT_MEDIA text, ACT_NOREC text, 
        ACT_OUTHOME text, ACT_PCARE text, ACT_PHYSICAL text, ACT_TRAVEL text, ACT_UNDOM text, ACT_WORK text);'''
        self.execute_with_new_connection(sql)

        conn = self.create_connection()
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                row = row[0].split(',')
                sql = '''INSERT INTO demographics_mtus VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'''
                conn.cursor().execute(sql, row)
        conn.close()

    def read_variables_and_variable_codes_into_table(self):
        sql = '''CREATE TABLE variables_mtus(variable_id text, description text);'''
        self.execute_with_new_connection(sql)
        sql = '''CREATE TABLE variable_codes_mtus(variable_id text, code text, value text);'''
        self.execute_with_new_connection(sql)

        demographic_codebook_path = 'resources/mtus/old/mtus_00007.xml'
        activity_codebook_path = 'resources/mtus/old/mtus_00008.xml'
        names = []
        labels = []
        categories = []
        for path in [demographic_codebook_path, activity_codebook_path]:
            with open(path, 'r') as f:
                file = f.read()
            soup = BeautifulSoup(file, 'lxml')
            for var in soup.find_all('var'):
                name = var['id']
                label = var.find('labl').contents
                label = label[0] if len(label) > 0 else None

                var_categories = {}
                for category in var.findAll('catgry'):
                    code = category.find('catvalu').contents
                    code = code[0] if len(code) > 0 else None
                    description = category.find('labl').contents
                    description = description[0] if len(description) > 0 else None
                    var_categories[code] = description
                if name not in names:
                    names.append(name)
                    labels.append(label)
                    categories.append(var_categories)

        # Variable identifier and name
        path_variables = 'resources/mtus/old/variables_mtus.csv'
        mode = 'a' if os.path.exists(path_variables) else 'w'
        conn = self.create_connection()
        with open(path_variables, mode, newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            headers = ["variable_id", "description"]
            writer.writerow(headers)
            for i in range(len(names)):
                row = [names[i], labels[i]]
                # 1) write to file
                writer.writerow(row)
                # 2) write to database
                conn.cursor().execute('''INSERT INTO variables_mtus VALUES (%s, %s)''', row)

        # Variable codes and code description
        path_variable_codes = 'resources/mtus/old/variable_codes_mtus.csv'
        mode = 'a' if os.path.exists(path_variables) else 'w'
        with open(path_variable_codes, mode, newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            headers = ["variable_id", "code", "value"]
            writer.writerow(headers)
            for i in range(len(names)):
                var_categories = categories[i]
                for key in var_categories.keys():
                    row = [names[i], key, var_categories[key]]
                    # 1) write to file
                    writer.writerow(row)
                    # 2) write to database
                    conn.cursor().execute('''INSERT INTO variable_codes_mtus VALUES (%s, %s, %s)''', row)
        conn.close()


def chunk_data(data, batch_size):
    """Yield successive batches of size batch_size from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
