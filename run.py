from pathlib import Path

import click
from src.database.database import Database
from src.config import GeneralConfig, RunConfig
from src.survey_processing.loading.travel_survey_encoding import SurveyEncoding
from src.mobility_generation.mobility_generator import MobilityGenerator
from src.survey_processing.loading.survey_loading import MTUSLoader

@click.command()
@click.option("--config_path", default='configs/config.toml', help="Path to config file")
@click.option("--run_config_path", default='configs/run.toml', help="Path to run config file")
@click.option("--config_id", default='1', help="Configuration identifier")
@click.option("--host", default='localhost', help="Database host")
@click.option("--port", default='5438', help="Database port")
@click.option("--database", default='astra', help="Database name")
@click.option("--user", default='user', help="Database user")
@click.option("--password", default='password', help="Database password")
@click.option("--nr_surveys", default='-1', help="Number of surveys to load from travel survey data set")
def create_ASTRA_trajectories(
        config_path: str,
        run_config_path: str,
        config_id: str,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        nr_surveys: int,
):
    db = Database(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )

    config_path = Path(config_path)
    run_config_path = Path(run_config_path)
    config_id = int(config_id)

    config = GeneralConfig.from_toml(config_path)
    run_config = RunConfig.from_file(run_config_path)

    survey_path = config.survey_processing_config.survey_path
    survey_paths = tuple([Path(survey_path)])

    survey_encoding = SurveyEncoding.from_config(encoding_path=config.encoding_path, seed=config.seed)
    loading_modules = tuple([MTUSLoader(
        config=config.survey_processing_config,
        encoding=survey_encoding,
        seed=config.seed)
    ])

    generator = MobilityGenerator(
        config=config,
        run_config=run_config,
        database=db,
        survey_encoding=survey_encoding,
        loading_modules=loading_modules,
        survey_paths=survey_paths,
    )

    db.create_astra_agents_table()
    db.create_astra_routes_checkin_table()
    db.create_config_grid_table()

    grid_id = db.get_grid_id(area=run_config.area, cell_size=config.tessellation_config.max_cell_size)
    db.append_config_grid(config_id=config_id, grid_id=grid_id)

    data_generator = generator.generate(run_config=run_config, config_id=config_id, nr_surveys=nr_surveys)
    print(len(list(data_generator)))


if __name__ == "__main__":
    create_ASTRA_trajectories()
