import affine
import modin.pandas as modin
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from typing import Callable, Iterable, Any, Sequence, Iterator
from rasterio.windows import from_bounds, Window
from shapely import Polygon, Point
from dataclasses import dataclass


def _get_prefix_distribution(
    path_population_characteristics: Path,
    categorical_persona_feature_names: Sequence[str],
    real_persona_feature_names: Sequence[str],
    area: Polygon,
) -> pd.DataFrame:
    """
    get distribution of persona feature from population characteristics file
    population characteristics file needs to have at least a timestamp column, a relative freq column
    and the specified persona features
    """
    persona_feature_names = list(categorical_persona_feature_names) + list(
        real_persona_feature_names
    )
    prefix_columns = persona_feature_names + [
        "timestamp",
        "relative_freq",
    ]

    prefix_df: modin.DataFrame = modin.read_parquet(path_population_characteristics)[
        prefix_columns
    ]
    prefix_df["relative_freq"] = (
        prefix_df["relative_freq"].to_numpy()
        / prefix_df["relative_freq"].to_numpy().sum()
    )

    for feature in categorical_persona_feature_names:
        prefix_df[feature] = prefix_df[feature].astype("category")

    prefix_df = prefix_df._to_pandas()

    return prefix_df


def _get_age_and_sex_structure(
    folder: Path,
    categorical_persona_feature_names: Sequence[str],
    real_persona_feature_names: Sequence[str],
    area: Polygon,
) -> pd.DataFrame:
    """
    get age and sex data of area from given folder
    """
    persona_feature_names = list(categorical_persona_feature_names) + list(real_persona_feature_names)
    # assert that AGE and SEX are the only two selected persona features
    assert (
        len(persona_feature_names) == 2
        and "AGE" in persona_feature_names
        and "SEX" in persona_feature_names
    )

    # traverse files for all age_sex structures
    data = list()
    for folder in folder.iterdir():
        # open geotiff
        with rasterio.open(
            folder,
            mode="r",
            driver="GTiff",
            transform=affine.identity,
        ) as dataset:
            # create window for reading geotiff
            min_x, min_y, max_x, max_y = area.bounds
            # takes format lat, lon
            window: Window = from_bounds(
                left=min_x,
                bottom=min_y,
                right=max_x,
                top=max_y,
                transform=dataset.transform,
            )

            # read file
            file_data = dataset.read(indexes=1, window=window, boundless=True, fill_value=np.nan)
            # get population count
            # todo: check nan stuff
            population_count = np.nansum(file_data) / 5
            sex = 1 if str(folder.name).split("_")[1] == "m" else 2
            age = int(str(folder.name).split("_")[2])
            for i in range(age, age + 5):
                data.append({"AGE": i, "SEX": sex, "relative_freq": population_count})
    data = pd.DataFrame(data=data)
    total_count = data["relative_freq"].sum()
    if total_count == 0:
        return pd.DataFrame()
    data["relative_freq"] = data["relative_freq"].apply(lambda x: x / total_count)
    data["SEX"] = data["SEX"].astype("category")
    return data


@dataclass(slots=True, frozen=True)
class Demographic:
    """Specifies demographic information of a certain area, such as age, sex distribution and nr of agents."""
    area: Polygon
    persona_feature_distribution: pd.DataFrame  # distribution of persona features
    nr_simulated_agents: int

    def select_agents_persona_features(self, rng: np.random.Generator) -> pd.DataFrame:
        """
        Select a persona feature for each agent that is to simulate by drawing randomly with likelihood according to
        the persona features' relative frequency.
        """
        if self.persona_feature_distribution.empty or self.nr_simulated_agents < 1:
            return pd.DataFrame()
        weights = self.persona_feature_distribution["relative_freq"].to_numpy()
        ids = rng.choice(a=len(weights), size=self.nr_simulated_agents, p=weights)
        return self.persona_feature_distribution.iloc[ids]

    @property
    def agent_density(self):
        area_size = self.area.area
        return self.nr_simulated_agents / area_size


class DemographicsData:
    """Maintenance and processing of the data sources of demographic information."""

    __data_paths: dict[Polygon, tuple[Path, Path]]  # dict storing a path for population counts and persona feature prefixes for the area key
    __categorical_persona_feature_names: Sequence[str]
    __real_persona_feature_names: Sequence[str]

    def __init__(
        self,
        data_paths: dict[Polygon, tuple[Path, Path]],
        categorical_persona_feature_names: Sequence[str],
        real_persona_feature_names: Sequence[str],
    ):
        self.__data_paths = data_paths
        self.__categorical_persona_feature_names = categorical_persona_feature_names
        self.__real_persona_feature_names = real_persona_feature_names

    def create_demographics(self, areas: Iterable[Polygon], nr_agents: int) -> Iterator[Demographic]:
        """
        Create a generator returning a Demographic object for each area. If nr_agents is negative, the number of
        agents is drawn from the actual population count.
        """
        population_count_per_area = []
        persona_distribution_per_area = []
        for area in areas:
            area_population_count = 0
            area_persona_distributions = []

            # for each data source file check whether its area overlaps with the area
            for area_covered_by_source, (persona_prefix_path, population_count_path) in self.get_demographic_data_area_and_paths(area):
                coordinates_population_count_dict = self.read_tiff_file(path=population_count_path, area=area_covered_by_source)
                population_count = sum(coordinates_population_count_dict.values())
                area_population_count += population_count

                persona_df = _get_age_and_sex_structure(
                    folder=persona_prefix_path,
                    categorical_persona_feature_names=self.__categorical_persona_feature_names,
                    real_persona_feature_names=self.__real_persona_feature_names,
                    area=area_covered_by_source,
                )
                if not persona_df.empty:
                    persona_df["relative_freq"] = persona_df["relative_freq"].apply(lambda x: x * population_count)
                area_persona_distributions.append(persona_df)

            population_count_per_area.append(area_population_count)
            area_persona_df = pd.concat(area_persona_distributions)

            # normalize relative frequencies
            if not area_persona_df.empty and area_persona_df["relative_freq"].sum() > 0:
                area_persona_df["relative_freq"] = (
                    area_persona_df["relative_freq"].to_numpy() / area_persona_df["relative_freq"].sum()
                )

            persona_distribution_per_area.append(area_persona_df)

        # distribute agents over areas according to the population distribution
        population_count_per_area = np.asarray(population_count_per_area)
        population_count_all_areas = population_count_per_area.sum()
        # if indicated here, create as many agents as in population
        if nr_agents < 0:
            nr_agents = np.floor(population_count_all_areas)
        distribution_simulated_agents_per_area = population_count_per_area * (nr_agents / population_count_all_areas)
        nr_simulated_agents_per_area = np.floor(distribution_simulated_agents_per_area)

        # distribute the remaining number of agents over the areas with the highest remainders
        remainder = distribution_simulated_agents_per_area - nr_simulated_agents_per_area
        remaining_nr_agents_to_simulate = nr_agents - int(nr_simulated_agents_per_area.sum())
        nr_simulated_agents_per_area[np.argsort(-remainder)[: remaining_nr_agents_to_simulate]] += 1
        nr_simulated_agents_per_area = np.asarray(nr_simulated_agents_per_area).astype(int)

        for area, nr_simulated_agents, persona_distribution in zip(areas, nr_simulated_agents_per_area, persona_distribution_per_area):
            yield Demographic(
                area=area,
                persona_feature_distribution=persona_distribution,
                nr_simulated_agents=nr_simulated_agents,
            )

    def get_demographic_data_area_and_paths(self, area: Polygon) -> Iterator[tuple[Polygon, tuple[Path, Path]]]:
        """Get source paths and the areas that are covered by the data sources indicated in config."""
        intersections = ((data_area.intersection(area), paths) for data_area, paths in self.__data_paths.items())
        for area_covered_by_data_source, paths in intersections:
            if not area_covered_by_data_source.is_empty:
                yield area_covered_by_data_source, paths

    def get_population_count_data(self, area: Polygon) -> dict[tuple, Any]:
        """Read population count data for area from files indicated in config."""
        population_count_data = dict()
        for intersection_polygon, (_, population_count_path) in self.get_demographic_data_area_and_paths(area=area):
            population_count_data.update(
                self.read_tiff_file(path=population_count_path, area=intersection_polygon)
            )

        return population_count_data

    def read_tiff_file(self, path: Path, area: Polygon) -> dict[tuple, Any]:
        """Read data from path, which should point to a geotiff file containing coordinates and a value."""
        with rasterio.open(path, mode="r", driver="GTiff", transform=affine.identity) as file:
            min_x, min_y, max_x, max_y = area.bounds
            window = rasterio.windows.from_bounds(left=min_x, bottom=min_y, right=max_x, top=max_y, transform=file.transform)
            data = file.read(indexes=1, window=window, boundless=True, fill_value=np.nan)

            coordinates_value = {}
            # saving sparse matrix as dict
            for i in range(int(data.shape[0])):
                for j in range(int(data.shape[1])):
                    lon, lat = file.xy(window.row_off + i, window.col_off + j)
                    population_density = data[i, j]
                    if area.contains(Point(lon, lat)) and not np.isnan(population_density):
                        coordinates_value[(lon, lat)] = np.max([0, population_density])

        return coordinates_value
