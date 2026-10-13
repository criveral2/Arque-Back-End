import pandas as pd

from app.domain.model_features import (
    CATEGORICAL_VARIABLES,
    NUMERIC_VARIABLES,
    TARGET_VARIABLE
)

from app.infrastructure.ml.pipelines import (
    build_random_forest_log_pipeline
)


class TrainProductionModelUseCase:

    def __init__(
        self,
        dataset_2023_repository,
        dataset_2025_repository,
        model_repository
    ):
        # Dataset histórico 2023.
        self.dataset_2023_repository = (
            dataset_2023_repository
        )

        # Dataset más reciente disponible, 2025.
        self.dataset_2025_repository = (
            dataset_2025_repository
        )

        # Repositorio encargado de guardar
        # el Pipeline entrenado.
        self.model_repository = (
            model_repository
        )

    def execute(self):

        # ---------------------------------------------------------
        # 1. CARGA DE DATASETS
        # ---------------------------------------------------------

        dataset_2023 = (
            self.dataset_2023_repository.load()
        )

        dataset_2025 = (
            self.dataset_2025_repository.load()
        )

        # ---------------------------------------------------------
        # 2. UNIÓN 2023 + 2025
        # ---------------------------------------------------------

        # Para la versión operativa de Arqué utilizamos
        # toda la información histórica disponible.
        dataset = pd.concat(
            [
                dataset_2023,
                dataset_2025
            ],
            ignore_index=True
        )

        # ---------------------------------------------------------
        # 3. VARIABLES DEL MODELO
        # ---------------------------------------------------------

        predictor_variables = (
            CATEGORICAL_VARIABLES
            + NUMERIC_VARIABLES
        )

        X = dataset[
            predictor_variables
        ].copy()

        y = dataset[
            TARGET_VARIABLE
        ].copy()

        # ---------------------------------------------------------
        # 4. CONSTRUCCIÓN DEL MODELO
        # ---------------------------------------------------------

        # Modelo seleccionado durante la fase experimental:
        #
        # Random Forest
        # +
        # transformación log1p(COSM2)
        model = (
            build_random_forest_log_pipeline()
        )

        # ---------------------------------------------------------
        # 5. ENTRENAMIENTO OPERATIVO
        # ---------------------------------------------------------

        # Ya no existe train/test split porque este modelo
        # tiene como objetivo ser utilizado por la aplicación.
        #
        # La evaluación científica ya se realizó previamente.
        model.fit(
            X,
            y
        )

        # ---------------------------------------------------------
        # 6. PERSISTENCIA DEL PIPELINE
        # ---------------------------------------------------------

        # Guardamos TODO el Pipeline:
        #
        # - imputación
        # - OneHotEncoder
        # - escalado
        # - Random Forest
        # - transformación logarítmica
        #
        # De esta manera la aplicación no debe volver
        # a preparar manualmente los datos.
        self.model_repository.save(
            model
        )

        # ---------------------------------------------------------
        # 7. RESULTADO
        # ---------------------------------------------------------

        return {
            "model": "Random Forest Log - Production",

            "training_years": [
                2023,
                2025
            ],

            "training_records": int(
                len(dataset)
            ),

            "predictor_variables": int(
                len(predictor_variables)
            ),

            "target_variable": (
                TARGET_VARIABLE
            ),

            "saved": True
        }