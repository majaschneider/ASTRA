import json
import torch
from datetime import datetime, timedelta
from typing import Iterator
from gluonts.dataset.common import TrainDatasets
from gluonts.torch import DeepAREstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from lightning.pytorch.callbacks import LearningRateFinder, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from src.config import (
    TrainingConfig,
    ModelConfig,
)
from src.survey_processing.dataclasses.agenda import Agenda
from src.tessellation.data_classes.demographic import Demographic
from src.survey_processing.utils.categorical_dist import (
    CategoricalOutput,
)
from src.survey_processing.transforms.dataset_adapter import DatasetAdapter


class SequenceModel:
    """
    class containing the activity model
    """

    __model: PyTorchPredictor  # deepar model
    dataset_adapter: (
        DatasetAdapter  # Adapter transforming agenda data into sequence model datasets
    )

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        dataset_adapter: DatasetAdapter,
        retrain_sequence_model: bool,
        training_datasets: TrainDatasets = None,
    ) -> None:

        model_path = training_config.checkpoint_dir
        self.dataset_adapter = dataset_adapter

        # load existing model if it exists and not specified otherwise
        if model_path.exists() and not retrain_sequence_model:
            self.__model = PyTorchPredictor.deserialize(path=model_path)
        else:
            # train model
            assert training_datasets is not None
            torch.set_float32_matmul_precision("high")

            # prepare categorical output distribution
            output_distribution = CategoricalOutput(
                beta=0.0,
                values=dataset_adapter.get_all_dataset_values(),
            )

            # define model
            model = DeepAREstimator(
                freq=f"{dataset_adapter.time_step_interval}T",
                prediction_length=training_datasets.metadata.prediction_length,
                context_length=len(dataset_adapter.persona_feature_names)
                ** int(dataset_adapter.prepend_persona_features),
                lr=training_config.learning_rate,
                num_layers=model_config.num_layers,
                hidden_size=model_config.hidden_size,
                dropout_rate=model_config.dropout_rate,
                weight_decay=training_config.optimizer_weight_decay,
                patience=training_config.scheduler_patience,
                distr_output=output_distribution,
                batch_size=training_config.batch_size,
                num_feat_static_cat=len(training_datasets.metadata.feat_static_cat),
                num_feat_static_real=len(training_datasets.metadata.feat_static_real),
                num_feat_dynamic_real=len(training_datasets.metadata.feat_dynamic_real),
                cardinality=[
                    int(static_categorical_feature.cardinality)
                    for static_categorical_feature in training_datasets.metadata.feat_static_cat
                ],
                scaling=False,
                num_parallel_samples=model_config.num_parallel_samples,
                num_batches_per_epoch=int(
                    len(training_datasets.train) / training_config.batch_size
                ),
                trainer_kwargs={
                    "max_epochs": training_config.max_epochs,
                    "profiler": "simple",
                    "logger": MLFlowLogger(
                        experiment_name="Activity model",
                        save_dir=str(training_config.path_logging),
                    ),
                    "callbacks": [
                        LearningRateFinder(),
                        EarlyStopping(
                            monitor="val_loss",
                            patience=training_config.early_stopping_patience,
                            min_delta=training_config.early_stopping_min_delta,
                        ),
                    ],
                },
            )
            # train model on training dataset, validate on test dataset
            self.__model = model.train(
                training_data=training_datasets.train,
                validation_data=training_datasets.test,
            )

            # save model to specified path
            self.__model.serialize(path=model_path)

            # adjust saved files to support custom output distribution
            predictor_parameters_path = model_path / "predictor.json"
            with predictor_parameters_path.open("r") as f:
                predictor_parameters = json.load(f)
            predictor_parameters["kwargs"]["prediction_net"]["kwargs"]["model_kwargs"][
                "distr_output"
            ]["kwargs"]["values"] = output_distribution.values.tolist()
            with predictor_parameters_path.open("w") as f:
                json.dump(predictor_parameters, f)

    def __call__(
        self,
        demographic: Demographic,
        starting_timestamp: datetime,
        starting_timestamp_std: timedelta,
    ) -> Iterator[Agenda]:
        """
        Generates synthetic agendas for a given demographic.
        """

        # create dataset from demographic
        dataset, persona_features = (
            self.dataset_adapter.create_dataset_from_demographic(
                demographic=demographic,
                starting_timestamp=starting_timestamp,
                starting_timestamp_std=starting_timestamp_std,
            )
        )

        # create time series forecast and transform to agenda
        for forecast, persona_feature_row in zip(
            self.__model.predict(dataset), persona_features.to_dict("records")
        ):
            yield self.dataset_adapter.inverse_transform_forecast(
                forecast=forecast, persona_features=persona_feature_row
            )
