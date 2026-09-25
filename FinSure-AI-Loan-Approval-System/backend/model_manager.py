# ============================================================
# FINSURE AI
# MODEL MANAGER
# PHASE 6
# ============================================================

from pathlib import Path

import joblib


class ModelManager:
    """
    Save and load trained models.
    """

    def __init__(
        self,
        models_directory="models",
    ) -> None:

        self.models_directory = Path(
            models_directory
        )

        self.models_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    def save_model(
        self,
        model,
        filename="best_model.pkl",
    ) -> Path:

        filepath = (
            self.models_directory
            / filename
        )

        joblib.dump(
            model,
            filepath,
        )

        return filepath

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(
        self,
        filename="best_model.pkl",
    ):

        filepath = (
            self.models_directory
            / filename
        )

        if not filepath.exists():

            raise FileNotFoundError(

                f"Model not found: {filepath}"

            )

        return joblib.load(
            filepath
        )

    # ========================================================
    # CHECK MODEL EXISTS
    # ========================================================

    def model_exists(
        self,
        filename="best_model.pkl",
    ) -> bool:

        filepath = (
            self.models_directory
            / filename
        )

        return filepath.exists()


# ============================================================
# END OF FILE
# ============================================================