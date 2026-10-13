import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from app.domain.model_features import (
    CATEGORICAL_VARIABLES,
    NUMERIC_VARIABLES,
    TARGET_VARIABLE
)

from app.infrastructure.ml.pipelines import (
    build_random_forest_log_pipeline
)


class EvaluateTemporalModelUseCase:

    def __init__(
        self,
        training_dataset_repository,
        test_dataset_repository
    ):
        # Dataset histórico utilizado para entrenar.
        # En nuestro experimento corresponde a ESED 2023.
        self.training_dataset_repository = (
            training_dataset_repository
        )

        # Dataset temporal posterior utilizado únicamente
        # para la evaluación final.
        # Corresponde a ESED 2025.
        self.test_dataset_repository = (
            test_dataset_repository
        )

    def execute(self):

        # ---------------------------------------------------------
        # 1. CARGA DE DATASETS
        # ---------------------------------------------------------

        training_dataset = (
            self.training_dataset_repository.load()
        )

        test_dataset = (
            self.test_dataset_repository.load()
        )

        predictor_variables = (
            CATEGORICAL_VARIABLES
            + NUMERIC_VARIABLES
        )

        # ---------------------------------------------------------
        # 2. DATOS DE ENTRENAMIENTO - ESED 2023
        # ---------------------------------------------------------

        # Utilizamos TODOS los registros preparados de 2023.
        X_train = training_dataset[
            predictor_variables
        ].copy()

        y_train = training_dataset[
            TARGET_VARIABLE
        ].copy()

        # ---------------------------------------------------------
        # 3. DATOS DE PRUEBA TEMPORAL - ESED 2025
        # ---------------------------------------------------------

        # No realizamos ninguna división adicional.
        #
        # Todo ESED 2025 funciona como conjunto
        # temporal de prueba.
        X_test = test_dataset[
            predictor_variables
        ].copy()

        y_test = test_dataset[
            TARGET_VARIABLE
        ].copy()

        # Conservamos los ID únicamente para poder
        # identificar posteriormente los mayores errores.
        test_ids = test_dataset["id"].copy()

        # ---------------------------------------------------------
        # 4. CONSTRUCCIÓN DEL MODELO SELECCIONADO
        # ---------------------------------------------------------

        # Random Forest con transformación log1p(COSM2),
        # seleccionado a partir de los experimentos realizados
        # exclusivamente con ESED 2023.
        model = build_random_forest_log_pipeline()

        # ---------------------------------------------------------
        # 5. ENTRENAMIENTO FINAL SOBRE 2023
        # ---------------------------------------------------------

        # El pipeline completo se ajusta únicamente con 2023:
        #
        # - imputación
        # - OneHotEncoder
        # - escalado
        # - Random Forest
        # - transformación logarítmica del objetivo
        model.fit(
            X_train,
            y_train
        )

        # ---------------------------------------------------------
        # 6. PREDICCIÓN SOBRE 2025
        # ---------------------------------------------------------

        # La predicción ya se devuelve automáticamente
        # en la escala original de USD/m².
        predictions = model.predict(
            X_test
        )

        # Error absoluto por registro.
        absolute_errors = np.abs(
            y_test.to_numpy()
            - predictions
        )

        # ---------------------------------------------------------
        # 7. MÉTRICAS DE EVALUACIÓN TEMPORAL
        # ---------------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        r2 = r2_score(
            y_test,
            predictions
        )

        # ---------------------------------------------------------
        # 8. DISTRIBUCIÓN DE ERRORES
        # ---------------------------------------------------------

        error_percentiles = {
            "50": float(
                np.percentile(
                    absolute_errors,
                    50
                )
            ),
            "75": float(
                np.percentile(
                    absolute_errors,
                    75
                )
            ),
            "90": float(
                np.percentile(
                    absolute_errors,
                    90
                )
            ),
            "95": float(
                np.percentile(
                    absolute_errors,
                    95
                )
            ),
            "99": float(
                np.percentile(
                    absolute_errors,
                    99
                )
            )
        }

        error_distribution = {
            "<=50": int(
                np.sum(
                    absolute_errors <= 50
                )
            ),

            "50-100": int(
                np.sum(
                    (absolute_errors > 50)
                    & (absolute_errors <= 100)
                )
            ),

            "100-200": int(
                np.sum(
                    (absolute_errors > 100)
                    & (absolute_errors <= 200)
                )
            ),

            "200-500": int(
                np.sum(
                    (absolute_errors > 200)
                    & (absolute_errors <= 500)
                )
            ),

            ">500": int(
                np.sum(
                    absolute_errors > 500
                )
            )
        }

        # ---------------------------------------------------------
        # 9. MAYORES ERRORES DE 2025
        # ---------------------------------------------------------

        results = X_test.copy()

        results["id"] = test_ids

        results["COSM2_REAL"] = (
            y_test
        )

        results["COSM2_PREDICTED"] = (
            predictions
        )

        results["ABSOLUTE_ERROR"] = (
            absolute_errors
        )

        largest_errors_df = (
            results
            .sort_values(
                "ABSOLUTE_ERROR",
                ascending=False
            )
            .head(20)
        )

        largest_errors = []

        for _, row in largest_errors_df.iterrows():

            largest_errors.append({
                "id": int(
                    row["id"]
                ),

                "COSM2_REAL": float(
                    row["COSM2_REAL"]
                ),

                "COSM2_PREDICTED": float(
                    row["COSM2_PREDICTED"]
                ),

                "ABSOLUTE_ERROR": float(
                    row["ABSOLUTE_ERROR"]
                ),

                "codcantf": int(
                    row["codcantf"]
                ),

                "NUPICAL": float(
                    row["NUPICAL"]
                ),

                "CSUTE": float(
                    row["CSUTE"]
                ),

                "CARCO": float(
                    row["CARCO"]
                )
            })

        # ---------------------------------------------------------
        # 10. RESULTADO FINAL
        # ---------------------------------------------------------

        return {
            "model": (
                "Random Forest Log - Temporal Evaluation"
            ),

            "training_year": 2023,
            "test_year": 2025,

            "training_records": int(
                len(X_train)
            ),

            "test_records": int(
                len(X_test)
            ),

            "predictor_variables": int(
                len(predictor_variables)
            ),

            "metrics": {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            },

            "error_percentiles": (
                error_percentiles
            ),

            "error_distribution": (
                error_distribution
            ),

            "largest_errors": (
                largest_errors
            )
        }