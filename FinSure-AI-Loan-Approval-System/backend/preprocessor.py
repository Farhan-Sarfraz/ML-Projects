# ============================================================
# FINSURE AI
# DATA PREPROCESSOR
# PHASE 3
# ============================================================

from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.preprocessing import (
    LabelEncoder,
)


class DataPreprocessor:
    """
    Professional preprocessing pipeline.

    Responsibilities
    --------------------------------------------------------
    1. Validate dataset
    2. Encode categorical features
    3. Split dataset
    4. Create train/test split
    5. Save preprocessing artifacts
    """

    def __init__(
        self,
        target_column: str = "Loan_Approved",
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:

        self.target_column = (
            target_column
        )

        self.test_size = (
            test_size
        )

        self.random_state = (
            random_state
        )

        self.label_encoder = (
            LabelEncoder()
        )

    # ========================================================
    # VALIDATE DATASET
    # ========================================================

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        if dataframe.empty:

            raise ValueError(
                "Dataset is empty."
            )

        if (
            self.target_column
            not in dataframe.columns
        ):

            raise ValueError(
                f"Target column "
                f"'{self.target_column}' "
                f"not found."
            )

    # ========================================================
    # ENCODE FEATURES
    # ========================================================

    def encode_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        if (
            "Employment_Status"
            in df.columns
        ):

            df[
                "Employment_Status"
            ] = self.label_encoder.fit_transform(
                df[
                    "Employment_Status"
                ]
            )

        return df

    # ========================================================
    # SPLIT FEATURES/TARGET
    # ========================================================

    def split_features_target(
        self,
        dataframe: pd.DataFrame,
    ):

        X = dataframe.drop(
            columns=[
                self.target_column
            ]
        )

        y = dataframe[
            self.target_column
        ]

        return X, y

    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    def create_train_test_split(
        self,
        X,
        y,
    ):

        return train_test_split(

            X,

            y,

            test_size=self.test_size,

            random_state=self.random_state,

            stratify=y,

        )

    # ========================================================
    # SAVE ARTIFACTS
    # ========================================================

    def save_artifacts(
        self,
        artifacts_path: str | Path,
    ) -> None:

        artifacts_path = Path(
            artifacts_path
        )

        artifacts_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(

            self.label_encoder,

            artifacts_path
            / "label_encoder.pkl",

        )

    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict[str, Any]:

        self.validate_dataset(
            dataframe
        )

        df = self.encode_features(
            dataframe
        )

        X, y = (
            self.split_features_target(
                df
            )
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.create_train_test_split(
            X,
            y,
        )

        return {

            "X_train":
            X_train,

            "X_test":
            X_test,

            "y_train":
            y_train,

            "y_test":
            y_test,

            "feature_names":
            list(
                X.columns
            ),

            "rows_train":
            len(X_train),

            "rows_test":
            len(X_test),

        }

    # ========================================================
    # INFO
    # ========================================================

    def info(
        self,
    ) -> Dict[str, Any]:

        return {

            "target_column":
            self.target_column,

            "test_size":
            self.test_size,

            "random_state":
            self.random_state,

        }


# ============================================================
# END OF FILE
# ============================================================