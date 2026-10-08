from pathlib import Path

import joblib

from app.domain.ports.model_repository import ModelRepository


class FileModelRepository(ModelRepository):

    def __init__(self, model_path: Path):
        self.model_path = model_path

    def exists(self) -> bool:
        return self.model_path.exists()

    def load(self):
        return joblib.load(self.model_path)

    def save(self, model) -> None:
        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            model,
            self.model_path
        )