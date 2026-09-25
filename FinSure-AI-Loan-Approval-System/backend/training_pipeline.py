# ============================================================
# FINSURE AI
# PRODUCTION TRAINING PIPELINE
# PHASE 9
# ============================================================

from pathlib import Path
from typing import Dict, Any

from backend.data_loader import DataLoader
from backend.preprocessor import DataPreprocessor
from backend.model_trainer import ModelTrainer
from backend.model_evaluator import ModelEvaluator
from backend.model_manager import ModelManager
from backend.artifact_manager import ArtifactManager


class TrainingPipeline:
    """
    Complete end-to-end training pipeline.
    """

    def __init__(
        self,
        dataset_path,
        models_directory="models",
        artifacts_directory="artifacts",
    ) -> None:

        self.dataset_path = Path(
            dataset_path
        )

        self.models_directory = Path(
            models_directory
        )

        self.artifacts_directory = Path(
            artifacts_directory
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self) -> Dict[str, Any]:

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        loader = DataLoader(
            self.dataset_path
        )

        df = loader.load_data()

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        preprocessor = (
            DataPreprocessor()
        )

        processed = (
            preprocessor.process(df)
        )

        # ----------------------------------------------------
        # SAVE PREPROCESSING ARTIFACTS
        # ----------------------------------------------------

        preprocessor.save_artifacts(
            self.artifacts_directory
        )

        # ----------------------------------------------------
        # TRAIN MODELS
        # ----------------------------------------------------

        trainer = ModelTrainer()

        training_result = (

            trainer.train(

                processed["X_train"],

                processed["y_train"],

                processed["X_test"],

                processed["y_test"],

            )

        )

        best_model = training_result[
            "best_model"
        ]

        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        evaluator = (
            ModelEvaluator()
        )

        evaluation = (

            evaluator.evaluate(

                model=best_model["model"],

                X_test=processed["X_test"],

                y_test=processed["y_test"],

                feature_names=processed[
                    "feature_names"
                ],

            )

        )

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        model_manager = (
            ModelManager(
                self.models_directory
            )
        )

        model_path = (

            model_manager.save_model(

                best_model["model"]

            )

        )

        # ----------------------------------------------------
        # SAVE FEATURE NAMES
        # ----------------------------------------------------

        artifact_manager = (
            ArtifactManager(
                self.artifacts_directory
            )
        )

        artifact_manager.save_artifact(

            processed[
                "feature_names"
            ],

            "feature_names.pkl",

        )

        # ----------------------------------------------------
        # SAVE ENCODER USING EXISTING PROJECT NAME
        # ----------------------------------------------------

        artifact_manager.save_artifact(

            preprocessor.label_encoder,

            "employment_encoder.pkl",

        )

        # ----------------------------------------------------
        # SAVE METADATA
        # ----------------------------------------------------

        metadata = {

            "best_model":
            best_model[
                "model_name"
            ],

            "accuracy":
            best_model[
                "metrics"
            ][
                "accuracy"
            ],

            "precision":
            best_model[
                "metrics"
            ][
                "precision"
            ],

            "recall":
            best_model[
                "metrics"
            ][
                "recall"
            ],

            "f1_score":
            best_model[
                "metrics"
            ][
                "f1_score"
            ],

            "roc_auc":
            best_model[
                "metrics"
            ][
                "roc_auc"
            ],

        }

        artifact_manager.save_artifact(

            metadata,

            "model_metadata.pkl",

        )

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return {

            "model_path":
            str(model_path),

            "best_model":
            best_model[
                "model_name"
            ],

            "accuracy":
            best_model[
                "metrics"
            ][
                "accuracy"
            ],

            "precision":
            best_model[
                "metrics"
            ][
                "precision"
            ],

            "recall":
            best_model[
                "metrics"
            ][
                "recall"
            ],

            "f1_score":
            best_model[
                "metrics"
            ][
                "f1_score"
            ],

            "roc_auc":
            best_model[
                "metrics"
            ][
                "roc_auc"
            ],

            "evaluation":
            evaluation,

        }


# ============================================================
# END OF FILE
# ============================================================