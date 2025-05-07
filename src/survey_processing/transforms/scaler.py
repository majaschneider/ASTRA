import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Iterable
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)
from sklearn.utils.validation import check_is_fitted

from src.config import SurveyProcessingConfig
from src.survey_processing.loading.travel_survey_encoding import (
    SurveyEncoding,
)


class Scaler:
    """
    data scaler for encoded activities and encoded persona features
    """

    save_path: Path
    agenda_scaler: MinMaxScaler | StandardScaler | RobustScaler
    persona_feature_scaler: dict[str, MinMaxScaler | StandardScaler | RobustScaler]
    persona_feature_categories: dict

    def __init__(
        self,
        survey_processing_config: SurveyProcessingConfig,
        survey_encoding: SurveyEncoding,
        reload: bool,
    ):
        self.save_path = survey_processing_config.path_scaler
        path = self.save_path / "scaler.pkl"
        # check for existing scalers
        if not reload and path.exists():
            with path.open("rb") as f:
                (
                    self.agenda_scaler,
                    self.persona_feature_scaler,
                    self.persona_feature_categories,
                ) = pickle.load(f)
        else:
            self.agenda_scaler = MinMaxScaler().fit(
                survey_encoding.all_encoded().reshape((-1, 1))
            )
            self.persona_feature_scaler, self.persona_feature_categories = (
                {},
                {},
            )  # TODO: easier encoding?
            for name in survey_processing_config.persona_feature_names:
                self.persona_feature_scaler[name] = MinMaxScaler((-2, -1))

    def inverse_transform_persona_feature_dict(
        self,
        persona_features: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            k: self.persona_feature_scaler[k].inverse_transform(
                np.full((1, 1), fill_value=v)
            )[0, 0]
            for k, v in persona_features.items()
        }

    def transform_persona_features(
        self,
        persona_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        scale each persona feature individually
        """
        for col in persona_features.columns:
            try:
                check_is_fitted(self.persona_feature_scaler[col])
                persona_features.loc[:, col] = (
                    self.persona_feature_scaler[col]
                    .transform(persona_features.loc[:, col].to_numpy().reshape((-1, 1)))
                    .ravel()
                )
            except NotFittedError:
                persona_features.loc[:, col] = (
                    self.persona_feature_scaler[col]
                    .fit_transform(
                        persona_features.loc[:, col].to_numpy().reshape((-1, 1))
                    )
                    .ravel()
                )

        return persona_features

    def to_categorical(
        self, df: pd.DataFrame, categorical_feature_names: Iterable[str]
    ) -> pd.DataFrame:
        """
        transform selected columns to categorical columns
        """

        for col in categorical_feature_names:
            if self.persona_feature_categories.get(col) is None:
                self.persona_feature_categories[col] = df[col].unique()
                df[col] = df[col].astype("category")
            else:
                df[col] = pd.Series(
                    pd.Categorical(
                        values=df[col], categories=self.persona_feature_categories[col]
                    )
                )

        return df

    def save(self):
        """
        save all scalers to disk
        """
        if (
            self.save_path is not None
            and self.agenda_scaler is not None
            and self.persona_feature_scaler is not None
        ):
            self.save_path.mkdir(parents=True, exist_ok=True)
            path = self.save_path / "scaler.pkl"
            with path.open("wb") as f:
                pickle.dump(
                    (
                        self.agenda_scaler,
                        self.persona_feature_scaler,
                        self.persona_feature_categories,
                    ),
                    f,
                    protocol=5,
                )

        return
