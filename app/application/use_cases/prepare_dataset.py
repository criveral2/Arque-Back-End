import pandas as pd

from app.domain.model_features import (
    CATEGORICAL_VARIABLES,
    NUMERIC_VARIABLES,
    TARGET_VARIABLE,
    FEATURE_SET_NAME
)


class PrepareDatasetUseCase:

    # Variable utilizada únicamente para mantener
    # trazabilidad con el registro original.
    #
    # NO se utilizará como predictor del modelo.
    TRACE_VARIABLES = [
        "id"
    ]

    def __init__(
        self,
        dataset_repository,
        prepared_dataset_repository
    ):
        # Repositorio encargado de leer
        # la base ESED original.
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

        # Cargamos siempre desde la fuente original
        # almacenada en data/raw.
        dataset = self.dataset_repository.load()

        # Guardamos el número inicial de registros
        # para mantener trazabilidad del proceso.
        original_records = len(dataset)

        # ---------------------------------------------------------
        # 2. DEFINICIÓN DE VARIABLES NECESARIAS
        # ---------------------------------------------------------

        # Las variables predictoras se obtienen directamente
        # desde model_features.py.
        #
        # Esto evita mantener diferentes listas de variables
        # en distintos archivos del proyecto.
        required_columns = (
            self.TRACE_VARIABLES
            + CATEGORICAL_VARIABLES
            + NUMERIC_VARIABLES
            + [TARGET_VARIABLE]
        )

        # ---------------------------------------------------------
        # 3. VALIDACIÓN DE VARIABLES
        # ---------------------------------------------------------

        # Comprobamos que todas las variables necesarias
        # existan realmente en la versión de ESED utilizada.
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
        # 4. SELECCIÓN DE VARIABLES
        # ---------------------------------------------------------

        # Conservamos únicamente:
        #
        # - id para trazabilidad
        # - variables predictoras
        # - variable objetivo COSM2
        prepared_dataset = dataset[
            required_columns
        ].copy()

        # ---------------------------------------------------------
        # 5. NORMALIZACIÓN DE VARIABLES NUMÉRICAS
        # ---------------------------------------------------------

        # Algunas variables numéricas de ESED pueden venir
        # almacenadas como texto.
        #
        # Ejemplo:
        #
        # "9"  -> 9
        # "3"  -> 3
        # " "  -> NaN
        #
        # errors="coerce" convierte cualquier valor que no
        # pueda interpretarse como número en NaN.
        #
        # Los NaN de las variables predictoras NO se eliminan.
        # Posteriormente serán tratados dentro del Pipeline
        # utilizando SimpleImputer.
        for column in NUMERIC_VARIABLES:
            prepared_dataset[column] = pd.to_numeric(
                prepared_dataset[column],
                errors="coerce"
            )

        # También garantizamos que la variable objetivo
        # COSM2 sea realmente numérica.
        prepared_dataset[TARGET_VARIABLE] = pd.to_numeric(
            prepared_dataset[TARGET_VARIABLE],
            errors="coerce"
        )

        # ---------------------------------------------------------
        # 6. DEPURACIÓN DE LA VARIABLE OBJETIVO
        # ---------------------------------------------------------

        # Conservamos únicamente registros donde COSM2
        # tenga un valor válido y mayor que cero.
        #
        # Los valores NaN también quedan excluidos
        # automáticamente mediante esta condición.
        #
        # En ESED 2023 se identificaron 582 registros
        # con COSM2 = 0 correspondientes a reconstrucciones.
        #
        # IMPORTANTE:
        # el archivo original ubicado en data/raw
        # nunca se modifica.
        prepared_dataset = prepared_dataset[
            prepared_dataset[TARGET_VARIABLE] > 0
        ].copy()

        # Número de registros después de la preparación.
        final_records = len(prepared_dataset)

        # ---------------------------------------------------------
        # 7. GUARDADO
        # ---------------------------------------------------------

        # Guardamos el nuevo dataset preparado
        # dentro de data/processed.
        self.prepared_dataset_repository.save(
            prepared_dataset
        )

        # ---------------------------------------------------------
        # 8. RESUMEN DEL PROCESO
        # ---------------------------------------------------------

        return {
            "feature_set": FEATURE_SET_NAME,

            "original_records": int(
                original_records
            ),

            "prepared_records": int(
                final_records
            ),

            "excluded_records": int(
                original_records - final_records
            ),

            "predictor_variables": int(
                len(CATEGORICAL_VARIABLES)
                + len(NUMERIC_VARIABLES)
            ),

            "categorical_variables": int(
                len(CATEGORICAL_VARIABLES)
            ),

            "numeric_variables": int(
                len(NUMERIC_VARIABLES)
            ),

            "target_variable": TARGET_VARIABLE
        }