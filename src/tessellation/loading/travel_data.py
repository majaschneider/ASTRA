import sys
from functools import partial
from itertools import combinations, pairwise
import datetime

import networkx as nx
import osmnx as ox
import numpy as np
from tqdm import tqdm
from shapely import Polygon
import pandas as pd

from src.database.database import Database
from src.tessellation.loading.osm_data import OSMData

class TravelData:

    grid_id: int

    area: Polygon

    cell_centroids: np.ndarray

    cell_ids: np.ndarray

    nr_cells: int

    database: Database

    # todo: change bicycle to bike and replace dict with list?
    transport_modes = {'drive': 'drive', 'bicycle': 'bike', 'walk': 'walk'}

    osm_data: OSMData

    speeds: dict[str, dict[str, float]]

    batch_size: int = 100

    range_to_calculate: list[float, float] = [0, 1]

    use_euclidean_travel: bool = False

    def __init__(
            self,
            grid_id: int,
            area: Polygon,
            cell_centroids: np.ndarray,
            cell_ids: np.ndarray,
            osm_data: OSMData,
            database: Database,
            speeds: dict[str, dict[str, float]],
            batch_size: int,
            range_to_calculate: list[float, float],
            use_euclidean_travel: bool,
    ):
        self.grid_id = grid_id
        self.area = area
        self.cell_centroids = cell_centroids
        self.cell_ids = cell_ids
        self.nr_cells = len(cell_centroids)
        self.osm_data = osm_data
        self.database = database
        self.speeds = speeds
        self.batch_size = batch_size
        self.range_to_calculate = range_to_calculate
        self.use_euclidean_travel = use_euclidean_travel

        self.database.create_travel_times_table(use_euclidean_travel=self.use_euclidean_travel)
        self.database.create_nearest_nodes_table()
        self.database.create_osmnx_graphs_table()

    def get_travel_times(self) -> dict:
        od_travel_times = {}
        # todo: first only direct connection to save time?
        for key in self.transport_modes.keys():
            transport_mode = self.transport_modes.get(key)

            # get travel times from database or calculate the range of origin destination node pairs indicated in config
            travel_times_for_mode = self.get_all_cells_travel_times(transport_mode=transport_mode)

            # if all data has been calculated
            if len(self.get_remaining_node_pairs_to_calculate(transport_mode=transport_mode, range_start=0, range_end=1)) == 0:
                # convert into origin-destination matrix for each pair of cells
                od_travel_times_mode = self.convert_to_origin_destination_matrix(
                    transport_mode=transport_mode,
                    travel_times=travel_times_for_mode
                )
            else:
                od_travel_times_mode = None

            od_travel_times[key] = od_travel_times_mode
        return od_travel_times

    def get_all_node_pairs(self, transport_mode: str) -> list:
        cell_id_to_node = self.get_nearest_nodes(transport_mode=transport_mode)
        unique_nodes = list(sorted(set(cell_id_to_node.values())))
        all_node_pairs = list(tuple(combinations(unique_nodes, 2)))
        return all_node_pairs

    def get_remaining_node_pairs_to_calculate(self, transport_mode: str, range_start: int, range_end: int) -> list:
        # number of required node pairs to calculate
        all_node_pairs = self.get_all_node_pairs(transport_mode=transport_mode)

        # limit to the range
        pairs_to_calculate = [int(np.floor(range_start * len(all_node_pairs))),
                              int(np.floor(range_end * len(all_node_pairs)))]
        all_node_pairs = all_node_pairs[pairs_to_calculate[0] : pairs_to_calculate[1]]

        # number of node pairs in database
        travel_times = self.database.get_travel_times(
            node_pairs=all_node_pairs,
            transport_mode=transport_mode,
            use_euclidean_travel=self.use_euclidean_travel
        )
        available_node_pairs = list(travel_times.keys())

        all_inputs_df = pd.DataFrame(data=np.asarray(all_node_pairs).reshape(-1, 2), columns=['origin', 'destination'])
        existing_inputs_df = pd.DataFrame(data=np.asarray(available_node_pairs).reshape(-1, 2), columns=['origin', 'destination'])
        remaining_inputs_df = all_inputs_df.merge(existing_inputs_df, on=['origin', 'destination'], how='left', indicator=True)
        remaining_inputs_df = remaining_inputs_df[remaining_inputs_df['_merge'] == 'left_only'][['origin', 'destination']]
        remaining_pairs_to_calculate = list(zip(remaining_inputs_df.origin.values, remaining_inputs_df.destination.values))

        return remaining_pairs_to_calculate

    def get_all_cells_travel_times(self, transport_mode: str) -> dict:
        """
        Get travel time in minutes between all pairwise coordinates. Travel times are queried from the database or
        if not existing, calculated and saved to the database. Calculate only the percentual range of node pairs
        indicated in config.
        """
        remaining_pairs_to_calculate = self.get_remaining_node_pairs_to_calculate(
            transport_mode=transport_mode,
            range_start=self.range_to_calculate[0],
            range_end=self.range_to_calculate[1]
        )
        print(datetime.datetime.now(), len(remaining_pairs_to_calculate), f'remaining node pairs to calculate ({transport_mode})')

        transportation_network = self.get_transportation_network(transport_mode=transport_mode)
        batches = list(chunk_data(data=remaining_pairs_to_calculate, batch_size=self.batch_size))
        for batch in tqdm(
                iterable=batches,
                total=len(batches),
                unit=f"{self.batch_size} cells",
                desc=f"Calculating travel times ({transport_mode})"
        ):
            self.calculate_and_save_travel_times(
                transportation_network=transportation_network,
                transport_mode=transport_mode,
                od_pairs=batch
            )

        travel_times = self.database.get_travel_times(
            node_pairs=self.get_all_node_pairs(transport_mode=transport_mode),
            transport_mode=transport_mode,
            use_euclidean_travel=self.use_euclidean_travel
        )

        return travel_times

    def convert_to_origin_destination_matrix(self, transport_mode: str, travel_times: dict) -> np.ndarray:
        nr_cells = len(self.cell_centroids)
        od_matrix = np.zeros(shape=(nr_cells, nr_cells))

        cell_id_to_node = self.get_nearest_nodes(transport_mode=transport_mode)
        for o_cell_idx in range(nr_cells):
            o_node = cell_id_to_node[self.cell_ids[o_cell_idx]]
            for d_cell_idx in range(nr_cells):
                d_node = cell_id_to_node[self.cell_ids[d_cell_idx]]
                if o_node != d_node:
                    (smaller_idx, larger_idx) = (o_node, d_node) if o_node < d_node else (d_node, o_node)
                    travel_time = travel_times[(smaller_idx, larger_idx)]
                    od_matrix[o_cell_idx][d_cell_idx] = travel_time
                    od_matrix[d_cell_idx][o_cell_idx] = travel_time

        return od_matrix

    def get_nearest_nodes(self, transport_mode: str) -> dict:
        """
        Get nearest nodes for all cells and the given transport mode from the database. If not existing, calculate from
        OSM and save to database. Return a dictionary with cell ids as keys and nearest nodes as values.
        """
        # get available data from database
        cell_id_to_node = self.database.get_nearest_nodes(
            grid_id=self.grid_id,
            transport_mode=transport_mode
        )

        # check for completeness
        calculated_cell_ids = list(cell_id_to_node.keys())
        cell_ids_to_calculate = [cell_id for cell_id in self.cell_ids if cell_id not in calculated_cell_ids]
        if len(cell_ids_to_calculate) > 0:
            # calculate nearest nodes that are not yet in the database
            cell_id_to_node = self.calculate_nearest_nodes(
                transport_mode=transport_mode,
                cell_ids_to_calculate=cell_ids_to_calculate,
            )

            # save data to database
            self.database.append_nearest_nodes(
                transport_mode=transport_mode,
                cell_to_nearest_nodes=cell_id_to_node
            )
        return cell_id_to_node

    def calculate_nearest_nodes(self, transport_mode: str, cell_ids_to_calculate: list) -> dict:
        cell_id_to_idx = {}
        for idx, cell_id in enumerate(self.cell_ids):
            cell_id_to_idx[cell_id] = idx

        cell_idxs_to_calculate = [cell_id_to_idx[cell_id] for cell_id in cell_ids_to_calculate]
        cell_centroids_to_calculate = self.cell_centroids[cell_idxs_to_calculate]

        # for each cell centroid calculate the osm node that is the closest
        nodes = ox.nearest_nodes(
            G=self.get_transportation_network(transport_mode=transport_mode),
            X=cell_centroids_to_calculate[:, 0].tolist(),
            Y=cell_centroids_to_calculate[:, 1].tolist(),
        )
        cell_id_to_node = dict(zip(cell_ids_to_calculate, nodes))
        return cell_id_to_node

    def get_transportation_network(self, transport_mode: str) -> nx.MultiDiGraph:
        # get from database
        transportation_network = self.database.get_osmnx_graph(
            grid_id=self.grid_id,
            transport_mode=transport_mode
        )

        # if not existing, calculate and save to database
        if transportation_network is None:
            transportation_network = self.osm_data.get_transportation_network(
                area=self.area,
                mode=transport_mode,
                default_speeds=self.speeds
            )
            if sys.getsizeof(transportation_network) >= 1e9:
                print('Transport network too large for database.')
            else:
                self.database.append_osmnx_graph(
                    grid_id=self.grid_id,
                    transport_mode=transport_mode,
                    graph=transportation_network
                )

        return transportation_network

    def calculate_and_save_travel_times(
            self,
            transportation_network: nx.MultiDiGraph,
            transport_mode: str,
            od_pairs: list[(int, int)]
    ):
        travel_times = {}
        for (origin, destination) in od_pairs:
            if self.use_euclidean_travel:
                travel_time = calculate_euclidean_distance_travel_time(
                    transportation_network=transportation_network,
                    origin_osm_id=origin,
                    destination_osm_id=destination,
                    speeds=self.speeds,
                    transport_mode=transport_mode,
                )
            else:
                travel_time = calculate_shortest_path_travel_time_with_manhattan_distance(
                    transportation_network=transportation_network,
                    origin_osm_id=origin,
                    destination_osm_id=destination,
                )
            (smaller_id, larger_id) = (origin, destination) if origin < destination else (destination, origin)
            travel_times[(smaller_id, larger_id)] = travel_time

        # save to database
        self.database.append_travel_times(
            transport_mode=transport_mode,
            travel_times=travel_times,
            use_euclidean_travel=self.use_euclidean_travel
        )


def calculate_euclidean_distance_travel_time(
        transportation_network: nx.MultiDiGraph,
        origin_osm_id: int,
        destination_osm_id: int,
        speeds: dict[str, dict[str, float]],
        transport_mode: str,
) -> float:
    coords = [0, 0]
    osm_id_pair = [
        (0, origin_osm_id),
        (1, destination_osm_id),
    ]
    for coords_idx, osm_node_id in osm_id_pair:
        val = transportation_network[osm_node_id]
        for i in range(3):
            if 'geometry' in list(val.keys()):
                coords[coords_idx] = val['geometry'].centroid
                break
            else:
                first_key = list(val.keys())[0]
                val = val[first_key]

    for i, osm_id in osm_id_pair:
        if coords[i] == 0:
            travel_time = np.inf
            return travel_time

    distance = ox.distance.great_circle(coords[0].y, coords[0].x, coords[1].y, coords[1].x) # meters
    avg_speed = np.average(list(speeds[transport_mode].values()))   # km/h
    travel_time = distance * 60 / (1_000 * avg_speed)    # minutes

    return travel_time


def calculate_shortest_path_travel_time_with_manhattan_distance(
        transportation_network: nx.MultiDiGraph,
        origin_osm_id: int,
        destination_osm_id: int,
) -> float:
    try:
        shortest_path = nx.astar_path(
            G=transportation_network,
            source=origin_osm_id,
            target=destination_osm_id,
            weight='travel_time',
            heuristic=partial(manhattan_distance, graph=transportation_network),
        )
        if np.unique(shortest_path).size == 1:
            travel_time = 0
        else:
            edge_travel_times = map(
                lambda x: transportation_network.get_edge_data(*x)[0]["travel_time"] / 60., # minutes
                pairwise(shortest_path),
            )
            travel_time = sum(edge_travel_times)
    except nx.exception.NetworkXNoPath:
        travel_time = np.inf
    return travel_time

def manhattan_distance(a: int, b: int, graph: nx.MultiDiGraph) -> float:
    x1, y1 = graph.nodes[a]["x"], graph.nodes[a]["y"]
    x2, y2 = graph.nodes[b]["x"], graph.nodes[b]["y"]
    return abs(x1 - x2) + abs(y1 - y2)


def chunk_data(data, batch_size):
    """Yield successive batches of size batch_size from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
