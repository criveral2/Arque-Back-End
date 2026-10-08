from pathlib import Path


class CsvPreparedDatasetRepository:

    def __init__(self, dataset_path: Path):
        # Ruta donde se almacenará el dataset preparado.
        self.dataset_path = dataset_path

    def save(self, dataset) -> None:

        # Crea la carpeta processed si todavía no existe.
        self.dataset_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Guarda el DataFrame como CSV.
        #
        # index=False evita agregar una columna adicional
        # con el índice interno de Pandas.
        #
        # sep=";" mantiene el mismo estilo de separación
        # utilizado por la base original de ESED.
        dataset.to_csv(
            self.dataset_path,
            sep=";",
            encoding="utf-8-sig",
            index=False
        )