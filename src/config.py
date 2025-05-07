import os

import shapely
from tomli import load
from datetime import datetime, timedelta
from typing import Iterable, Any
import attr
from pathlib import Path

import osmnx as ox
from shapely import box, Polygon, is_valid
from shapely.validation import explain_validity


def _valid_polygon(instance, attribute, value) -> None:
    """
    check if given value is a valid polygon
    """
    assert value is None or isinstance(value, Polygon)
    if value is not None and not is_valid(value):
        raise ValueError(
            f"Value {value} is not a valid polygon: {explain_validity(value)}"
        )
    return


def custom_validator(instance, attribute, value):
    if not (value >= 1 or value == -1):
        raise ValueError(f"{attribute.name} must be >= 1 or == -1, got {value}")


@attr.s(slots=True, frozen=True)
class ModelConfig:
    """
    config for activity model
    """

    hidden_size: int = attr.field(
        validator=[
            attr.validators.instance_of(int),
            attr.validators.ge(1),
        ],
    )
    num_layers: int = attr.field(
        validator=[attr.validators.instance_of(int), attr.validators.ge(1)]
    )
    dropout_rate: float = attr.field(
        validator=[attr.validators.instance_of(float), attr.validators.ge(0.0)]
    )
    num_parallel_samples: int = attr.field(
        validator=[attr.validators.instance_of(int), attr.validators.ge(1)]
    )  # number of predicted samples to draw from

    @classmethod
    def from_section(cls, section: dict[str, Any]) -> "ModelConfig":
        return cls(
            hidden_size=int(section["hidden_size"]),
            num_layers=int(section["num_layers"]),
            dropout_rate=float(section["dropout_rate"]),
            num_parallel_samples=int(section["num_parallel_samples"]),
        )


@attr.s(slots=True, frozen=True)
class TrainingConfig:
    """
    config for training activity model
    """

    batch_size: int = attr.field(
        validator=[attr.validators.instance_of(int), attr.validators.ge(1)]
    )

    learning_rate: float = attr.field(
        validator=[attr.validators.instance_of(float), attr.validators.ge(1e-9)]
    )
    optimizer_weight_decay: float = attr.field(
        validator=attr.validators.instance_of(float)
    )

    scheduler_patience: int = attr.field(
        validator=[attr.validators.instance_of(int), attr.validators.ge(1)]
    )

    early_stopping_min_delta: float = attr.field(
        validator=attr.validators.instance_of(float)
    )
    early_stopping_patience: int = attr.field(
        validator=[attr.validators.instance_of(int), attr.validators.ge(1)]
    )

    path_logging: Path = attr.field(validator=attr.validators.instance_of(Path))
    max_epochs: int = attr.field(
        validator=[attr.validators.instance_of(int), attr.validators.ge(1)]
    )
    checkpoint_dir: Path = attr.field(validator=attr.validators.instance_of(Path))

    @classmethod
    def from_section(cls, section: dict[str, Any]) -> "TrainingConfig":
        return cls(
            learning_rate=float(section["learning_rate"]),
            batch_size=int(section["batch_size"]),
            checkpoint_dir=Path(section["sequence_model_dir"]),
            max_epochs=int(section["max_epochs"]),
            path_logging=Path(section["path_logging"]),
            optimizer_weight_decay=float(section["optimizer_weight_decay"]),
            scheduler_patience=int(section["scheduler_patience"]),
            early_stopping_min_delta=float(section["early_stopping_min_delta"]),
            early_stopping_patience=int(section["early_stopping_patience"]),
        )


@attr.s(slots=True, frozen=True)
class SurveyProcessingConfig:
    """
    config for processing travel survey data
    """
    survey_path: str = attr.field()
    time_step_interval: int = attr.field(
        validator=[attr.validators.instance_of(int), attr.validators.ge(1)]
    )  # time step size
    travel_survey_sample_size: int = attr.field(
        validator=[attr.validators.instance_of(int), custom_validator]
    )
    test_size: float = attr.field(
        validator=[
            attr.validators.instance_of(float),
            attr.validators.ge(0.0),
            attr.validators.lt(1.0),
        ]
    )
    prepend_persona_features: bool = attr.field(
        validator=attr.validators.instance_of(bool)
    )  # prepend persona features for synthetic agenda sequence_model
    path_scaler: Path = attr.field(validator=attr.validators.instance_of(Path))

    dummy_date: datetime = attr.field()

    categorical_persona_feature_names: list[str] = attr.field(
        validator=attr.validators.instance_of(list), default=attr.Factory(list)
    )
    real_persona_feature_names: list[str] = attr.field(
        validator=attr.validators.instance_of(list), default=attr.Factory(list)
    )

    @classmethod
    def from_section(
        cls, section: dict, accepted_bool_operators: Iterable[Any]
    ) -> "SurveyProcessingConfig":
        return cls(
            survey_path=section["survey_path"],
            time_step_interval=int(section["scanning_interval"]),
            test_size=float(section["test_size"]),
            travel_survey_sample_size=int(section["travel_survey_sample_size"]),
            prepend_persona_features=section["prepend_persona_features"]
            in accepted_bool_operators,
            path_scaler=Path(section["scaler_folder"]),
            dummy_date=datetime(
                year=2020,
                month=int(section["dummy_month"]),
                day=int(section["dummy_day"]),
            ),
            categorical_persona_feature_names=section["categorical_persona_features"],
            real_persona_feature_names=section["real_persona_features"],
        )

    @property
    def persona_feature_names(self) -> list[str]:
        return self.categorical_persona_feature_names + self.real_persona_feature_names


@attr.define(slots=True, frozen=True)
class TessellationConfig:
    """
    config describing the tesselation of data
    """

    max_cell_size: float = attr.field(
        validator=[attr.validators.instance_of(float), attr.validators.ge(1e-3)]
    )  # max size of cell. Determines how much data is loaded at once

    speeds: dict[str, dict[str, float]] = attr.field(
        validator=attr.validators.instance_of(dict)
    )  # default speeds for each travel mode

    # if True, travel times are calculated from Euclidean distance, otherwise from the shortest path
    use_euclidean_travel: bool = attr.field()

    travel_time_folder: Path = attr.field()
    # clustering_parameters: dict = attr.field(default=attr.Factory(dict))

    @classmethod
    def from_section(
        cls, section: dict[str, Any], path_default_speeds: Path
    ) -> "TessellationConfig":

        with path_default_speeds.open("rb") as f:
            config = load(f)

        return cls(
            max_cell_size=float(section["max_cell_size"]),
            # clustering_parameters=dict(section["clustering_parameters"]),
            travel_time_folder=Path(section["travel_time_folder"]),
            use_euclidean_travel=bool(section["use_euclidean_travel"]),
            speeds={
                "walk": {k: float(v) for k, v in config["walk"].items()},
                "bike": {k: float(v) for k, v in config["bike"].items()},
                "drive": {k: float(v) for k, v in config["drive"].items()},
            },
        )


@attr.s(slots=True, frozen=True)
class RouteCreationConfig:
    """Configuration concerning route generation."""
    max_travel_percentage_error: float = attr.field(
        validator=[
            attr.validators.instance_of(float),
            attr.validators.gt(0.0),
            attr.validators.lt(1.0),
        ]
    )

    gamma: float = attr.field(
        validator=[
            attr.validators.instance_of(float),
            attr.validators.ge(0.0),
            attr.validators.le(1.0),
        ]
    )

    rho: float = attr.field(
        validator=[
            attr.validators.instance_of(float),
            attr.validators.ge(0.0),
            attr.validators.le(1.0),
        ]
    )

    alpha: float = attr.field(
        validator=[
            attr.validators.instance_of(float),
            attr.validators.le(1.0),
            attr.validators.ge(0.0),
        ]
    )

    eta: float = attr.field(validator=attr.validators.instance_of(float))

    embed_persona_features: bool = attr.field()

    allow_returns: bool = attr.field()

    use_weighted_average_search: bool = attr.field()
    weight_impact_on_average: bool = attr.field()

    weight_explore_poi_type_frequency: float = attr.field()
    weight_explore_semantic_similarity: float = attr.field()
    weight_explore_gravity: float = attr.field()

    weight_return_frequency: float = attr.field()
    weight_return_recency: float = attr.field()
    weight_return_semantic_similarity: float = attr.field()

    return_min_similarity_score: float = attr.field()

    top_n: int = attr.field()

    @classmethod
    def from_section(
        cls, section: dict[str, Any], accepted_bool_operators: Iterable[str]
    ) -> "RouteCreationConfig":
        return cls(
            gamma=float(section["gamma"]),
            rho=float(section["rho"]),
            alpha=float(section["alpha"]),
            eta=float(section["eta"]),
            embed_persona_features=section["embed_persona_features"] in accepted_bool_operators,
            allow_returns=section['allow_returns'],
            max_travel_percentage_error=float(section["max_travel_percentage_error"]),
            use_weighted_average_search=section["use_weighted_average_search"] in accepted_bool_operators,
            weight_impact_on_average=section["weight_impact_on_average"] in accepted_bool_operators,
            weight_explore_poi_type_frequency=float(section["weight_explore_poi_type_frequency"]),
            weight_explore_semantic_similarity=float(section["weight_explore_semantic_similarity"]),
            weight_explore_gravity=float(section["weight_explore_gravity"]),
            weight_return_frequency=float(section["weight_return_frequency"]),
            weight_return_recency=float(section["weight_return_recency"]),
            weight_return_semantic_similarity=float(section["weight_return_semantic_similarity"]),
            return_min_similarity_score=float(section["return_min_similarity_score"]),
            top_n=int(section["top_n"])
        )


@attr.define(slots=True)
class RunConfig:
    """
    general options for a simulation
    """
    area: Polygon = attr.field(validator=_valid_polygon)
    num_agents: int = attr.field(validator=attr.validators.instance_of(int))
    starting_datetime: datetime = attr.field(validator=attr.validators.instance_of(datetime))
    save_path: Path = attr.field(validator=attr.validators.instance_of(Path))
    starting_datetime_std: timedelta = (attr.field())  # std for creating normally distributed deviations from base starting time stamp
    batch_size: int = (attr.field())
    range_to_calculate: list[float, float] = (attr.field())

    @classmethod
    def from_file(cls, path: Path) -> "RunConfig":
        with path.open("rb") as f:
            config = load(f)

        try:
            area = config['area']
            str_tuples = area.replace('],[', ';').replace(']', '').replace('[', '').split(';')
            area_list = []
            for el in str_tuples:
                el = el.split(',')
                area_list.append([float(el[0]), float(el[1])])
            config['area'] = Polygon(area_list)
        except Exception:
            city_gdf = ox.geocode_to_gdf(config['area'])
            city_polygon = city_gdf.iloc[0].geometry
            if isinstance(city_polygon, shapely.MultiPolygon):
                city_polygon = shapely.convex_hull(city_polygon)
            config['area'] = city_polygon

        return cls(
            area=config["area"],
            num_agents=int(config["num_agents"]),
            starting_datetime=datetime_from_config_section(section=config["starting_datetime"]),
            starting_datetime_std=timedelta_from_config_section(section=config["starting_datetime_std"]),
            save_path=Path(config["save_path"]),
            batch_size=int(config["batch_size"]),
            range_to_calculate=config["range_to_calculate"],
        )


@attr.define(frozen=True, slots=True)
class GeneralConfig:
    seed: int = attr.field()

    # dataset is only reloaded when sequence model is retrained
    force_reload_dataset: bool = attr.field(validator=attr.validators.instance_of(bool))
    retrain_sequence_model: bool = attr.field(validator=attr.validators.instance_of(bool))

    survey_processing_config: SurveyProcessingConfig = attr.field()
    encoding_path: Path = attr.field()

    model_config: ModelConfig = attr.field()

    training_config: TrainingConfig = attr.field()

    tessellation_config: TessellationConfig = attr.field()

    route_creation_config: RouteCreationConfig = attr.field()

    path_processed_dataset: Path = attr.field()
    path_embedding_index: Path = attr.field(validator=attr.validators.instance_of(Path))
    paths_demographic_data: dict[Polygon, tuple[Path, Path]] = attr.field(validator=attr.validators.instance_of(dict))
    folder_poi_descriptions: Path = attr.field()
    path_excluded_pois: Path = attr.field()
    path_default_speeds: Path = attr.field()

    num_cpus: int = attr.field(
        validator=[
            attr.validators.instance_of(int),
            attr.validators.ge(1),
            attr.validators.le(os.cpu_count()),
        ]
    )

    @classmethod
    def from_toml(cls, config_path: Path) -> "GeneralConfig":
        accepted_bool_operators = (True, "y", "yes", "on", 1, "true", "True")
        with config_path.open("rb") as f:
            config = load(f)

        general_section = config["general"]
        survey_processing_config = SurveyProcessingConfig.from_section(
            section=config["survey_processing"],
            accepted_bool_operators=accepted_bool_operators,
        )
        model_config = ModelConfig.from_section(section=config["sequence_model"])
        training_config = TrainingConfig.from_section(section=config["training"])
        tessellation_config = TessellationConfig.from_section(
            section=config["tessellation"],
            path_default_speeds=Path(general_section["path_default_speeds"])
        )
        optimization_config = RouteCreationConfig.from_section(
            section=config["route_creation"],
            accepted_bool_operators=accepted_bool_operators,
        )


        bbox_demographic_paths = general_section["bbox_demographic_paths"]
        bbox_demographic_paths = (
            (
                data["covered_area"],
                (Path(data["prefix_data"]), Path(data["population_count_data"])),
            )
            for data in bbox_demographic_paths
        )
        bbox_demographic_paths = {
            box(
                xmin=parsed_bbox[0][0],
                ymin=parsed_bbox[0][1],
                xmax=parsed_bbox[1][0],
                ymax=parsed_bbox[1][1],
            ): paths
            for parsed_bbox, paths in bbox_demographic_paths
        }

        return cls(
            seed=int(general_section["seed"]),
            force_reload_dataset=general_section["force_reload_dataset"]
            in accepted_bool_operators,
            retrain_sequence_model=general_section["retrain_sequence_model"]
            in accepted_bool_operators,
            encoding_path=Path(general_section["encoding_path"]),
            path_processed_dataset=Path(general_section["path_processed_dataset"]),
            path_embedding_index=Path(general_section["path_embedding_index"]),
            paths_demographic_data=bbox_demographic_paths,
            folder_poi_descriptions=Path(general_section["folder_poi_descriptions"]),
            path_excluded_pois=Path(general_section["path_excluded_pois"]),
            path_default_speeds=Path(general_section["path_default_speeds"]),
            survey_processing_config=survey_processing_config,
            model_config=model_config,
            training_config=training_config,
            tessellation_config=tessellation_config,
            route_creation_config=optimization_config,
            num_cpus=int(general_section["num_cpus"]),
        )


def datetime_from_config_section(section: dict) -> datetime:
    return datetime(
        year=section["year"],
        month=section["month"],
        day=section["day"],
        hour=section["hour"],
        minute=section["minute"],
    )


def timedelta_from_config_section(section: dict) -> timedelta:
    return timedelta(
        days=int(section["days"]),
        hours=int(section["hours"]),
        minutes=int(section["minutes"]),
    )
