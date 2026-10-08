from typing import Protocol


class PreparedDatasetRepository(Protocol):

    # Define el contrato que debe cumplir cualquier repositorio
    # encargado de guardar un dataset ya preparado.
    def save(self, dataset) -> None:
        ...