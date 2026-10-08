from pathlib import Path

import pandas as pd

from app.domain.ports.dataset_repository import DatasetRepository


class EsedCsvRepository(DatasetRepository):

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def exists(self) -> bool:
        return self.dataset_path.exists()

    def load(self):
        return pd.read_csv(
            self.dataset_path,
            sep=";",
            encoding="utf-8-sig",
            low_memory=False
        )