import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split

from app.domain.model_features import (
    CATEGORICAL_VARIABLES,
    NUMERIC_VARIABLES,
    TARGET_VARIABLE
)

from app.infrastructure.ml.pipelines import (
    build_random_forest_log_pipeline
)


class AnalyzeRandomForestLogUseCase:

    def __init__(self, dataset_repository):

        # Repositorio utilizado para cargar
        # ESED 2023 preparado.
        self.dataset_repository = dataset_repository

    def execute(self):

        # ---------------------------------------------------------
        # 1. CARGA DEL DATASET
        # ---------------------------------------------------------

        dataset = self.dataset_repository.load()

        predictor_variables = (
            CATEGORICAL_VARIABLES
            + NUMERIC_VARIABLES
        )

        # Variables predictoras.
        X = dataset[
            predictor_variables
        ].copy()

        # Variable objetivo.
        y = dataset[
            TARGET_VARIABLE
        ].copy()

        # Conservamos el ID únicamente para poder
        # identificar los registros con mayores errores.
        ids = dataset["id"].copy()

        # ---------------------------------------------------------
        # 2. DIVISIÓN TRAIN / VALIDATION
        # ---------------------------------------------------------

        # Utilizamos exactamente la misma división
        # de todos los experimentos anteriores.
        (
            X_train,
            X_validation,
            y_train,
            y_validation,
            id_train,
            id_validation
        ) = train_test_split(
            X,
            y,
            ids,
            test_size=0.20,
            random_state=42
        )

        # ---------------------------------------------------------
        # 3. ENTRENAMIENTO
        # ---------------------------------------------------------

        # Random Forest entrenado utilizando
        # log1p(COSM2) como variable objetivo interna.
        model = build_random_forest_log_pipeline()

        model.fit(
            X_train,
            y_train
        )

        # ---------------------------------------------------------
        # 4. PREDICCIONES
        # ---------------------------------------------------------

        # Las predicciones regresan automáticamente
        # a la escala original USD/m² mediante expm1().
        predictions = model.predict(
            X_validation
        )

        # Error absoluto de cada registro.
        absolute_errors = np.abs(
            y_validation.to_numpy()
            - predictions
        )

        # ---------------------------------------------------------
        # 5. MÉTRICAS GENERALES
        # ---------------------------------------------------------

        mae = mean_absolute_error(
            y_validation,
            predictions
        )

        rmse = mean_squared_error(
            y_validation,
            predictions
        ) ** 0.5

        r2 = r2_score(
            y_validation,
            predictions
        )

        # ---------------------------------------------------------
        # 6. PERCENTILES DEL ERROR
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

        # ---------------------------------------------------------
        # 7. DISTRIBUCIÓN DE LOS ERRORES
        # ---------------------------------------------------------

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
        # 8. IMPORTANCIA DE VARIABLES
        # ---------------------------------------------------------

        # Permutation Importance mide cuánto empeora
        # el MAE cuando se desordena cada predictor.
        #
        # Cuanto mayor sea el valor, mayor es el aporte
        # de esa variable al modelo.
        importance = permutation_importance(
            model,
            X_validation,
            y_validation,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )

        feature_importance = []

        for variable, importance_value, std_value in zip(
            predictor_variables,
            importance.importances_mean,
            importance.importances_std
        ):
            feature_importance.append({
                "variable": variable,
                "importance": float(
                    importance_value
                ),
                "std": float(
                    std_value
                )
            })

        # Ordenamos desde la variable más importante.
        feature_importance = sorted(
            feature_importance,
            key=lambda item: item["importance"],
            reverse=True
        )

        # ---------------------------------------------------------
        # 9. REGISTROS CON MAYORES ERRORES
        # ---------------------------------------------------------

        validation_results = (
            X_validation.copy()
        )

        validation_results["id"] = (
            id_validation
        )

        validation_results["COSM2_REAL"] = (
            y_validation
        )

        validation_results[
            "COSM2_PREDICTED"
        ] = predictions

        validation_results[
            "ABSOLUTE_ERROR"
        ] = absolute_errors

        # Seleccionamos los 20 errores más grandes.
        largest_errors_df = (
            validation_results
            .sort_values(
                "ABSOLUTE_ERROR",
                ascending=False
            )
            .head(20)
        )

        # Solo devolvemos algunas variables principales
        # para que la respuesta sea legible.
        largest_errors = []

        for _, row in largest_errors_df.iterrows():

            largest_errors.append({
                "id": int(row["id"]),
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
                "CTIPOBR": int(
                    row["CTIPOBR"]
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
        # 10. RESULTADO
        # ---------------------------------------------------------

        return {
            "model": "Random Forest Log Diagnostic",

            "validation_records": int(
                len(X_validation)
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

            "feature_importance": (
                feature_importance
            ),

            "largest_errors": (
                largest_errors
            )
        }