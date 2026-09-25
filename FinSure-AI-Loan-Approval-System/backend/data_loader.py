# ============================================================
# FINANCIAL LOAN APPROVAL SYSTEM
# DATA LOADER
# PHASE 1
# ============================================================

from pathlib import Path

import pandas as pd


class DataLoader:

    """
    Loads loan dataset.
    """

    def __init__(
        self,
        file_path: str | Path,
    ) -> None:

        self.file_path = Path(
            file_path
        )

    def load_data(
        self,
    ) -> pd.DataFrame:

        if not self.file_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        dataframe = pd.read_csv(
            self.file_path
        )

        return dataframe

    def dataset_info(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:

        return {

            "rows": dataframe.shape[0],

            "columns": dataframe.shape[1],

            "missing_values":
            dataframe.isnull().sum().sum(),

        }


# ============================================================
# END OF FILE
# ============================================================