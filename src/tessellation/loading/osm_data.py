from typing import Iterable, Iterator
from collections import Counter
from itertools import pairwise

from pathlib import Path
from tomli import load
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from shapely import Polygon, MultiPolygon, Point, box
from shapely.geometry.base import BaseGeometry
import ray

from src.database.database import Database
from src.mobility_generation.encoding.embedding_collection import EmbeddingCollection
from src.tessellation.data_classes.poi_set import POISet
from src.tessellation.data_classes.pois import Poi


class OSMData:

    map_features: Iterable[str] = ["amenity", "building", "office", "shop", "tourism", "leisure", "sport"]

    excluded_pois: dict[str, tuple[str]]

    poi_descriptions: dict[str, dict[str, dict[str, str]] | dict[str, str]]

    embedding_collection: EmbeddingCollection

    nr_subregions_per_axis: int

    database: Database

    def __init__(
            self,
            description_dir: Path,
            path_excluded_pois: Path,
            num_sub_regions: int,
            embedding_collection: EmbeddingCollection,
            map_features: Iterable[str] | None,
            database: Database,
    ):
        self.database = database
        self.database.create_osm_data_table()

        # overwrite map_features if they were provided
        if map_features is not None:
            self.map_features = map_features

        poi_descriptions = dict()
        for path in description_dir.iterdir():
            with path.open("rb") as f:
                poi_descriptions.update(dict(load(f)))
        self.poi_descriptions = poi_descriptions

        with path_excluded_pois.open("rb") as f:
            excluded_pois = dict(load(f))
        self.excluded_pois = excluded_pois

        self.nr_subregions_per_axis = int(np.floor(np.sqrt(num_sub_regions)))

        self.embedding_collection = embedding_collection

    def get_osm_gdf(self, cell_id: int) -> gpd.GeoDataFrame | None:
        """
        Get the area's map features from the database. If they don't exist, request the data from
        Openstreetmap and save it to the database.
        """
        osm_gdf = self.database.get_osm_data(cell_id=cell_id)
        if osm_gdf is None:
            area = self.database.get_cell_area(cell_id=cell_id)
            osm_gdf = self.request_area(area=area, map_features=self.map_features)
            osm_gdf = self.clean(osm_gdf)
            osm_gdf = self.preprocess(osm_gdf)
            if osm_gdf is None:
                # no data available but to remember that it has been requested from osm save an empty gdf
                osm_gdf = gpd.GeoDataFrame()
            self.database.append_osm_data(cell_id=cell_id, osm_gdf=osm_gdf, is_empty=osm_gdf.empty)

        if osm_gdf.empty:
            osm_gdf = None

        return osm_gdf

    def request_area(self, area: Polygon, map_features: Iterable[str] = None) -> gpd.GeoDataFrame | None:
        """Request the area's map features from Openstreetmap."""
        if map_features is None:
            map_features = self.map_features
        tags = {tag: True for tag in map_features}
        try:
            osm_poi_gdf = ox.features_from_polygon(polygon=area, tags=tags)
        except Exception as e:
            osm_poi_gdf = None
        if osm_poi_gdf is not None and osm_poi_gdf.empty:
            osm_poi_gdf = None
        return osm_poi_gdf

    def clean(self, poi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
        if poi_gdf is None:
            return None

        # Remove rows where all map features are missing
        columns_with_map_feature = [col for col in poi_gdf.columns if col in self.map_features]
        poi_gdf = poi_gdf.dropna(how="all", subset=columns_with_map_feature)

        # Remove rows where map features contain POIs that are to be excluded
        for column in columns_with_map_feature:
            if column in self.excluded_pois.keys():
                excluded_rows = poi_gdf[column].apply(lambda x: x in self.excluded_pois[column])
                poi_gdf = poi_gdf.loc[~excluded_rows]
        if 'name' not in poi_gdf.columns:
            poi_gdf = None
        if poi_gdf is not None and poi_gdf.empty:
            poi_gdf = None
        return poi_gdf

    def preprocess(self, poi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
        if poi_gdf is None:
            return None

        # transform geometry to centroid
        poi_gdf["centroid"] = poi_gdf["geometry"].astype(object).apply(calculate_centroid)

        # assign centroid as new geometry
        poi_gdf = gpd.GeoDataFrame(poi_gdf, geometry='centroid', crs=4326)

        return poi_gdf

    def get_transportation_network(
            self,
            area: MultiPolygon | Polygon,
            mode: str,
            default_speeds: dict[str, dict[str, float]]
    ) -> nx.MultiDiGraph:
        """
        get transportation network of area by combining cell transportation networks
        """
        try:
            g = ox.graph_from_polygon(
                polygon=area,
                network_type=mode,
                retain_all=True,
                truncate_by_edge=True,
            )
        except ValueError:
            return None
        if mode == "drive":
            g = ox.add_edge_speeds(g, hwy_speeds=default_speeds["drive"])
        else:
            for u, v, k, data in g.edges(keys=True, data=True):
                if isinstance(data["highway"], Iterable):
                    road_types = (
                        r for r in data["highway"] if r in default_speeds[mode].keys()
                    )
                    road_speeds = [default_speeds[mode][r] for r in road_types]
                    if len(road_speeds) < 1:
                        road_speeds = [default_speeds[mode]["default"]]
                    data["speed_kph"] = float(np.median(road_speeds))
                else:
                    data["speed_kph"] = default_speeds[mode].get(
                        data["highway"], default_speeds[mode]["default"]
                    )
        return ox.add_edge_travel_times(g)

    def transform_pois_to_sentences(self, poi_gdf: gpd.GeoDataFrame) -> tuple[str, ...]:
        """Transforms each POI in poi_gdf into the shape: 'At {category} ({section}). {poi_subclass}: {description}'."""
        # Filter columns to requested map features only
        map_feature_columns = [col for col in poi_gdf.columns if col in self.map_features]
        df = poi_gdf[map_feature_columns]

        representations = list()
        # todo
        for row in df.itertuples(index=False):
            non_nan_attributes = (
                (column, value)
                for column, value in zip(df.columns, row)
                if not (isinstance(value, float) and np.isnan(value) and value)
            )

            poi_representation_parts = list()
            for i, (category, poi_subclass) in enumerate(non_nan_attributes, start=1):
                for section in get_sections(
                    element=poi_subclass,
                    poi_descriptions=self.poi_descriptions[category],
                ):
                    if section is None:
                        # get description if available
                        description = self.poi_descriptions[category].get(
                            poi_subclass, ""
                        )

                        if poi_subclass == "yes":
                            representation_part = f"At/In {category}: {description}"
                        elif category == "office":
                            representation_part = (
                                f"At office ({poi_subclass.lower()}): {description}"
                            )
                        else:
                            representation_part = f"At/In {poi_subclass}: {description}"
                    else:
                        description = self.poi_descriptions[category][section][
                            poi_subclass
                        ]
                        if poi_subclass == "yes":
                            representation_part = f"At/In {category}: {description}"
                        elif category == "shop":
                            representation_part = f"At/In shop ({section.lower()}). {poi_subclass}: {description}"
                        elif section == "other":
                            representation_part = f"At/In {poi_subclass}: {description}"
                        else:
                            representation_part = f"At place for {section.lower()}. {poi_subclass}: {description}"

                    poi_representation_parts.append(
                        representation_part.replace("_", " ")
                    )

            representations.append(str(" ; ".join(set(poi_representation_parts))))

        return tuple(representations)

    def calculate_spatial_density(self, poi_gdf: gpd.GeoDataFrame) -> np.ndarray[float, ...]:
        """
        Based on all POIs in this gdf create a grid over the area covering all POIs and calculate the spatial density of
        POIS in each cell. POIs are then assigned the spatial density of the cell they are in.
        """
        if len(poi_gdf) == 1:
            densities_per_point = np.asarray([1])
            return densities_per_point

        # centroid of each POI's geometry (POI geometries can be points or areas)
        gdf = gpd.GeoDataFrame(data={'geometry': [Point((p.x, p.y)) for p in poi_gdf["centroid"]]}, geometry='geometry', crs=4326)
        # Reproject to a projected CRS (e.g., UTM) that preserves distances
        projected_crs = gdf.estimate_utm_crs()
        gdf_proj = gdf.to_crs(projected_crs)
        poi_centroids = np.asarray([(point.x, point.y) for point in gdf_proj["geometry"]])

        xs = np.linspace(
            start=poi_centroids[:, 0].min(),
            stop=poi_centroids[:, 0].max(),
            num=self.nr_subregions_per_axis + 1,
        )
        ys = np.linspace(
            start=poi_centroids[:, 1].min(),
            stop=poi_centroids[:, 1].max(),
            num=self.nr_subregions_per_axis + 1,
        )

        # for each subregion calculate whether a POI is contained and the cell's surface area
        points_per_cell = [
            (
                ((poi_centroids >= (x_min, y_min)) & (poi_centroids <= (x_max, y_max))).all(axis=1),    # truth values over all POIs with regards to the current subregion
                box(xmin=x_min, ymin=y_min, xmax=x_max, ymax=y_max).area,
            )
            for x_min, x_max in pairwise(xs)
            for y_min, y_max in pairwise(ys)
        ]

        # calculate spatial density of each cell as number of POIs divided by the overlap of all POIs' convex hull's surface area with a subregion
        densities_per_cell_and_point = np.asarray(
            [is_in_cell * (is_in_cell.sum() / area) for is_in_cell, area in points_per_cell if area > 0],   # sum() -> nr of POIs
            dtype=float
        )
        densities_per_point = np.sum(densities_per_cell_and_point, axis=0)

        return densities_per_point

    def transform_osm_pois_to_poi_data(self, poi_gdf: gpd.GeoDataFrame) -> np.ndarray[Poi]:
        poi_sentences = self.transform_pois_to_sentences(poi_gdf=poi_gdf)
        relevance_scores = self.calculate_spatial_density(poi_gdf=poi_gdf)

        pois = np.asarray([
            Poi(
                relevance_score=float(relevance_scores[i]),
                longitude=poi_gdf['centroid'].iloc[i].x,
                latitude=poi_gdf['centroid'].iloc[i].y,
                poi_sentence=str(poi_sentences[i]),
                poi_name=poi_gdf['name'].iloc[i],
            )
            for i in range(len(poi_gdf))
        ])
        return pois

    def transform_to_poi_set(self, poi_gdf: gpd.GeoDataFrame | None, embedding_collection: EmbeddingCollection) -> POISet | None:
        """
        Transformation of an OSM POI Geo-dataframe into a reduced representation, called POISet, that contains
        statistical information about the POIs and their embedding.
        """
        if poi_gdf is None:
            return None
        # Get centroid coordinates of all POIs
        coordinates = np.asarray([(p.x, p.y) for p in poi_gdf["centroid"].tolist()], dtype=float)

        centroid = np.mean(coordinates, axis=0)

        poi_sentences = self.transform_pois_to_sentences(poi_gdf)
        counter = Counter(poi_sentences)
        unique_poi_sentence_frequencies = np.fromiter(counter.values(), dtype=int)
        unique_poi_sentences = tuple(counter.keys())

        # calculate mean distance from all POIs to the POI collection's centroid
        mean_distance_to_centroid = np.mean(np.linalg.norm(coordinates - centroid, axis=1))

        # index all poi sentences and retrieve their index ids
        embedding_ids = ray.get(embedding_collection.get_ids.remote(unique_poi_sentences))

        poi_set = POISet(
            centroid_lon=float(centroid[0]),
            centroid_lat=float(centroid[1]),
            poi_sentences=unique_poi_sentences,
            poi_sentence_frequencies=unique_poi_sentence_frequencies,
            mean_distance_to_centroid=mean_distance_to_centroid,
            embedding_ids=embedding_ids
        )

        return poi_set



def calculate_centroid(geometry: BaseGeometry) -> Point:
    if type(geometry) is not Point:
        geometry = geometry.centroid
    return geometry


def get_sections(element: str, poi_descriptions: dict[str, dict[str, str]] | dict[str, str]) -> Iterator[str]:
    """Get description section of POI."""
    returned_element = False
    if element in poi_descriptions:
        # descriptions have no sub sections
        yield None

    for section, values in poi_descriptions.items():
        if isinstance(values, dict) and element in values:
            returned_element = True
            yield section

    if not returned_element:
        # no section found
        yield None
