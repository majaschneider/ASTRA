import numpy as np
import ray
import shapely
from ray import ObjectRef
from shapely import Polygon, box
from tqdm import tqdm

from src.config import TessellationConfig, RunConfig, GeneralConfig
from src.database.database import Database
from src.tessellation.data_classes.demographic import DemographicsData
from src.tessellation.data_classes.poi_set import POISet
from src.tessellation.data_classes.pois import Poi
from src.tessellation.loading.osm_data import OSMData
from src.tessellation.loading.travel_data import TravelData
from src.mobility_generation.encoding.embedding_collection import EmbeddingCollection
from src.mobility_generation.utils.scoring import calculate_od_gravity_scores
import geopandas as gpd


class Tessellation:
    """
    Create a grid formed tessellation that partitions the simulation region into even cells. For efficiency, travel
    times and population counts are only calculated and stored for cells that contain POIs.
    """
    database: Database

    tessellation_config: TessellationConfig
    run_config: RunConfig

    embedding_collection: EmbeddingCollection
    travel_data: TravelData

    area: Polygon
    cells: np.ndarray
    grid: gpd.GeoDataFrame
    grid_id: int
    cell_ids_with_pois: np.ndarray
    cell_ids: np.ndarray

    # The following data structures store data only for cells that contain POIs and can be linked to cells via cell_ids_with_pois
    cells_with_pois: np.ndarray
    cell_centroids_with_pois: np.ndarray
    poi_sets_with_pois: np.ndarray
    od_travel_times: dict[str, ObjectRef]
    od_gravity_scores: np.ndarray
    population_counts: np.ndarray

    def __init__(
            self,
            area: Polygon,
            embedding_collection: EmbeddingCollection,
            demographic_data: DemographicsData,
            tessellation_config: TessellationConfig,
            run_config: RunConfig,
            config: GeneralConfig,
            database: Database
    ):
        self.database = database
        self.tessellation_config = tessellation_config
        self.run_config = run_config
        self.config = config
        self.area = area
        self.embedding_collection = embedding_collection

        self.create_grid()

        self.osm_data = OSMData(
            description_dir=self.config.folder_poi_descriptions,
            path_excluded_pois=self.config.path_excluded_pois,
            num_sub_regions=64, # todo: make parameter
            embedding_collection=self.embedding_collection,
            map_features=None,
            database=self.database
        )

        self.calculate_tessellation_data(demographic_data)

    def create_grid(self, crs="EPSG:4326"):
        self.database.create_grid_table()
        self.database.create_cells_table()
        self.database.create_cell_grid_relation_table()

        cell_size = self.tessellation_config.max_cell_size
        self.grid = calculate_grid(polygon=self.area, cell_size=cell_size, crs=crs)
        self.cells = np.asarray(list(self.grid.geometry))

        # Get or create grid-id from database
        grid_id = self.database.get_grid_id(area=self.area, cell_size=cell_size)
        if grid_id is None:
            self.grid_id = self.database.append_grid(area=self.area, cell_size=cell_size)
        else:
            self.grid_id = grid_id

        # Get or create cell-ids and cell-grid-relations from database
        cell_ids = []
        for cell in self.cells.tolist():
            cell_id = self.database.get_cell_id(area=cell)
            if cell_id is None:
                cell_id = self.database.append_cell(area=cell)
            cell_ids.append(cell_id)
            self.database.append_cell_grid_relation(grid_id=self.grid_id, cell_id=cell_id)
        self.cell_ids = np.asarray(cell_ids)


    def calculate_tessellation_data(self, demographic_data: DemographicsData) -> None:
        # load poi information per cell from OSM for all cells
        osm_poi_gdfs = self.calculate_pois()

        # transform pois of each cell in tessellation to reduced representation (=poi set), a poi_set can be None
        poi_sets = self.calculate_poi_sets(osm_poi_gdfs)

        # filter those poi sets and cells where pois exist
        self.cell_ids_with_pois = np.asarray([self.cell_ids[i] for i, poi_set in enumerate(poi_sets) if poi_set is not None])
        self.cell_centroids_with_pois = np.asarray([(poi_set.centroid_lon, poi_set.centroid_lat) for poi_set in poi_sets if poi_set is not None])
        self.cells_with_pois = np.asarray([self.cells[i] for i, poi_set in enumerate(poi_sets) if poi_set is not None])
        self.poi_sets_with_pois = np.asarray([ray.put(poi_sets[i]) for i, poi_set in enumerate(poi_sets) if poi_set is not None])

        self.travel_data = TravelData(
            grid_id=self.grid_id,
            area=self.area,
            cell_centroids=self.cell_centroids_with_pois,
            cell_ids=self.cell_ids_with_pois,
            osm_data=self.osm_data,
            database=self.database,
            speeds=self.tessellation_config.speeds,
            batch_size=self.run_config.batch_size,
            range_to_calculate=self.run_config.range_to_calculate,
            use_euclidean_travel=self.tessellation_config.use_euclidean_travel
        )

        # todo: make df iso matrix
        # calculate pairwise travel times between all points for each transport mode
        od_travel_times = self.travel_data.get_travel_times()
        # put to object store
        self.od_travel_times = {}
        for key in od_travel_times.keys():
            self.od_travel_times[key] = ray.put(od_travel_times.get(key))

        # get population counts of each non-empty cell
        self.population_counts = np.fromiter(
            (sum(demographic_data.get_population_count_data(cell).values()) for cell in self.cells_with_pois),
            dtype=float,
        )

        # calculate gravity scores for each origin-destination cell pair
        self.od_gravity_scores = calculate_od_gravity_scores(
            locations=self.cell_centroids_with_pois,
            relevances=self.population_counts
        )

    def calculate_pois(self):
        osm_poi_gdfs = []
        for cell_id in tqdm(
                iterable=self.cell_ids,
                total=len(self.cell_ids),
                unit="cell",
                desc="Loading OSM Pois",
        ):
            osm_poi_gdf = self.osm_data.get_osm_gdf(cell_id)
            osm_poi_gdfs.append(osm_poi_gdf)
        return osm_poi_gdfs

    def calculate_poi_sets(self, osm_poi_gdfs) -> list[POISet]:
        poi_sets = []
        for osm_poi_gdf in tqdm(
                iterable=osm_poi_gdfs,
                total=len(osm_poi_gdfs),
                unit="osm_poi_gdf",
                desc="Transforming OSM data from cells to Poi Sets",
        ):
            poi_set = self.osm_data.transform_to_poi_set(
                poi_gdf=osm_poi_gdf,
                embedding_collection=self.embedding_collection,
            ) if osm_poi_gdf is not None else None
            poi_sets.append(poi_set)
        return poi_sets

    def get_poi_set(self, idx: int) -> POISet:
        """
        get cluster with given centroid coordinates
        """
        poi_set = ray.get(self.poi_sets_with_pois[idx])
        return poi_set


    def get_travel_times(self, origin_cell_idx: int, destination_cell_idxs: np.ndarray | None, travel_mode: str) -> np.ndarray:
        """Get travel times between an origin cell and all destination cells."""
        return ray.get(self.od_travel_times[travel_mode])[origin_cell_idx, destination_cell_idxs]

    def get_cell_poi_data(self, cell_idx: int) -> np.ndarray[Poi]:
        """
        get exact poi data of cell with given index
        """
        osm_gdf = self.osm_data.get_osm_gdf(cell_id=int(self.cell_ids_with_pois[cell_idx]))
        poi_data = self.osm_data.transform_osm_pois_to_poi_data(osm_gdf)
        return poi_data

    def get_reachable_cell_idxs(
        self,
        origin_cell_idx: int,
        travel_mode: str,
        travel_time_minutes: float,
        max_error: float,
    ) -> np.ndarray:
        """
        Get the ids of all cells that are reachable from the origin cell in the given travel time and using the given
        travel mode while allowing a certain deviation from the actual travel time by max_error.
        """
        # get travel times from origin cell to all other cells in the tesselation
        od_travel_times = self.get_travel_times(
            origin_cell_idx=origin_cell_idx,
            destination_cell_idxs=None,
            travel_mode=travel_mode
        ).ravel()

        # calculate deviation of travel times from given travel time
        travel_time_errors = np.asarray(
            [abs(travel_time_minutes - od_travel_time) for od_travel_time in od_travel_times]
        )

        # get ids of cells for which the travel time error is below the max_error threshold
        reachable_cell_idxs = np.where(travel_time_errors <= max_error)[0]

        return reachable_cell_idxs

    def to_gdf(self):
        data = list(zip(self.cells_with_pois, self.population_counts))
        gdf = gpd.GeoDataFrame(data=data, columns=['area', 'population'], geometry='area', crs="EPSG:4326")
        return gdf

def calculate_grid(polygon: shapely.Polygon, cell_size: float, crs: str = "EPSG:4326"):
    # Create a GeoDataFrame from the input polygon
    gdf = gpd.GeoDataFrame([{"geometry": polygon}], crs=crs)

    # Reproject to a projected CRS (e.g., UTM) that preserves distances
    projected_crs = gdf.estimate_utm_crs()
    gdf_proj = gdf.to_crs(projected_crs)

    # Get the bounds of the polygon in projected coordinates
    minx, miny, maxx, maxy = gdf_proj.total_bounds

    # Generate the grid coordinates
    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)

    # Create grid cells
    cells = []
    for x in x_coords:
        for y in y_coords:
            cell = box(x, y, x + cell_size, y + cell_size)
            # Check if the cell intersects the polygon
            if gdf_proj.geometry.iloc[0].intersects(cell):
                cells.append(cell)

    cells = [gdf_proj.geometry.iloc[0].intersection(grid_cell) for grid_cell in cells]
    cells = np.asarray([a for a in cells if not a.is_empty])

    # Create a GeoDataFrame for the grid
    grid = gpd.GeoDataFrame({"geometry": cells}, crs=projected_crs)

    # Reproject back to the original CRS
    grid = grid.to_crs(crs)

    return grid
