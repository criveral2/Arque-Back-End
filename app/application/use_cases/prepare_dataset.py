class PrepareDatasetUseCase:

    # Variable utilizada únicamente para mantener
    # trazabilidad con el registro original.
    #
    # NO se utilizará como predictor del modelo.
    TRACE_VARIABLES = [
        "id"
    ]

    # Variables categóricas candidatas.
    CATEGORICAL_VARIABLES = [
        "codcantf",
        "careaur",
        "CTIPOBR",
        "cimi",
        "piso",
        "estru",
        "pared",
        "cubi",
        "CASAEDIF",
        "CDISPUSO"
    ]

    # Variables numéricas candidatas.
    NUMERIC_VARIABLES = [
        "NUPICAL",
        "CSUTE",
        "CARCO"
    ]

    # Variable que intentaremos predecir.
    TARGET_VARIABLE = "COSM2"

    def __init__(
        self,
        dataset_repository,
        prepared_dataset_repository
    ):
        # Repositorio que lee ESED original.
        self.dataset_repository = dataset_repository

        # Repositorio encargado de guardar
        # el dataset preparado.
        self.prepared_dataset_repository = (
            prepared_dataset_repository
        )

    def execute(self):

        # ---------------------------------------------------------
        # 1. CARGA DE DATOS
        # ---------------------------------------------------------

        # Cargamos siempre desde data/raw.
        dataset = self.dataset_repository.load()

        original_records = len(dataset)

        # ---------------------------------------------------------
        # 2. VALIDACIÓN DE VARIABLES
        # ---------------------------------------------------------

        required_columns = (
            self.TRACE_VARIABLES
            + self.CATEGORICAL_VARIABLES
            + self.NUMERIC_VARIABLES
            + [self.TARGET_VARIABLE]
        )

        # Comprobamos que todas las columnas necesarias
        # realmente existan en ESED.
        missing_columns = [
            column
            for column in required_columns
            if column not in dataset.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Variables no encontradas: {missing_columns}"
            )

        # ---------------------------------------------------------
        # 3. SELECCIÓN DE VARIABLES
        # ---------------------------------------------------------

        # Trabajamos únicamente con las variables relevantes
        # para nuestra investigación.
        prepared_dataset = dataset[
            required_columns
        ].copy()

        # ---------------------------------------------------------
        # 4. DEPURACIÓN DE LA VARIABLE OBJETIVO
        # ---------------------------------------------------------

        # Eliminamos del conjunto de entrenamiento aquellos
        # registros donde COSM2 no contiene un costo válido.
        #
        # En ESED 2023 se identificaron 582 registros COSM2 = 0,
        # correspondientes a reconstrucciones.
        #
        # El dataset original NO se modifica.
        prepared_dataset = prepared_dataset[
            prepared_dataset[self.TARGET_VARIABLE] > 0
        ].copy()

        final_records = len(prepared_dataset)

        # ---------------------------------------------------------
        # 5. GUARDADO
        # ---------------------------------------------------------

        # Guardamos el dataset preparado en data/processed.
        self.prepared_dataset_repository.save(
            prepared_dataset
        )

        # ---------------------------------------------------------
        # 6. RESUMEN DEL PROCESO
        # ---------------------------------------------------------

        return {
            "original_records": int(original_records),
            "prepared_records": int(final_records),
            "excluded_records": int(
                original_records - final_records
            ),
            "predictor_variables": int(
                len(self.CATEGORICAL_VARIABLES)
                + len(self.NUMERIC_VARIABLES)
            ),
            "target_variable": self.TARGET_VARIABLE
        }