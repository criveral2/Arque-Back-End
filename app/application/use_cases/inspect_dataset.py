class InspectDatasetUseCase:

    def __init__(self, dataset_repository):
        self.dataset_repository = dataset_repository

    def execute(self):
        if not self.dataset_repository.exists():
            raise FileNotFoundError(
                "No se encontró el dataset ESED."
            )

        dataset = self.dataset_repository.load()

        return {
            "rows": dataset.shape[0],
            "columns": dataset.shape[1],
            "column_names": dataset.columns.tolist(),
            "duplicates": int(dataset.duplicated().sum()),
            "null_values": int(dataset.isnull().sum().sum())
        }