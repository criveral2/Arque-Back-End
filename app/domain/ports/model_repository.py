from typing import Protocol


class ModelRepository(Protocol):

    def exists(self) -> bool:
        ...

    def load(self):
        ...

    def save(self, model) -> None:
        ...