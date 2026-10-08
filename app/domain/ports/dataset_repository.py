from typing import Protocol


class DatasetRepository(Protocol):

    def exists(self) -> bool:
        ...

    def load(self):
        ...