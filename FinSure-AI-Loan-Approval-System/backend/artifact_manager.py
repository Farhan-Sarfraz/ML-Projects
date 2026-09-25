# ============================================================
# FINSURE AI
# ARTIFACT MANAGER
# PHASE 8
# ============================================================

from pathlib import Path
import joblib


class ArtifactManager:
    """
    Save and load preprocessing artifacts.
    """

    def __init__(
        self,
        artifact_directory="artifacts",
    ) -> None:

        self.artifact_directory = Path(
            artifact_directory
        )

        self.artifact_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # SAVE ARTIFACT
    # ========================================================

    def save_artifact(
        self,
        artifact,
        filename,
    ):

        filepath = (
            self.artifact_directory
            / filename
        )

        joblib.dump(
            artifact,
            filepath,
        )

        return filepath

    # ========================================================
    # LOAD ARTIFACT
    # ========================================================

    def load_artifact(
        self,
        filename,
    ):

        filepath = (
            self.artifact_directory
            / filename
        )

        if not filepath.exists():

            raise FileNotFoundError(
                f"{filepath} not found."
            )

        return joblib.load(
            filepath
        )

    # ========================================================
    # INFO
    # ========================================================

    def info(
        self,
    ):

        files = list(
            self.artifact_directory.glob(
                "*"
            )
        )

        return {

            "artifact_count":
            len(files),

            "directory":
            str(
                self.artifact_directory
            ),
        }


# ============================================================
# END OF FILE
# ============================================================