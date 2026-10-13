from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from math import sqrt

from app.application.use_cases.prepare_training_data import (
    PrepareTrainingDataUseCase
)

from app.infrastructure.ml.pipelines import (
    build_random_forest_pipeline
)


class AnalyzeRandomForestUseCase:

    def __init__(self, dataset_repository):

        # Repositorio que permite cargar
        # ESED 2023 previamente preparado.
        self.dataset_repository = dataset_repository

    def execute(self):

        # ---------------------------------------------------------
        # 1. CARGA DEL DATASET
        # ---------------------------------------------------------

        # Cargamos el dataset completo porque posteriormente
        # utilizaremos el id original para rastrear los registros
        # con mayores errores.
        dataset = self.dataset_repository.load()

        # ---------------------------------------------------------
        # 2. PREPARACIÓN DE ENTRENAMIENTO / VALIDACIÓN
        # ---------------------------------------------------------

        # Utilizamos exactamente la misma división 80/20
        # de los experimentos anteriores.
        training_data_use_case = PrepareTrainingDataUseCase(
            self.dataset_repository
        )

        training_data = training_data_use_case.execute()

        X_train = training_data["X_train"]
        X_validation = training_data["X_validation"]

        y_train = training_data["y_train"]
        y_validation = training_data["y_validation"]

        # ---------------------------------------------------------
        # 3. ENTRENAMIENTO DEL RANDOM FOREST BASE
        # ---------------------------------------------------------

        # Utilizamos exactamente la configuración base
        # que obtuvo el mejor MAE en nuestros experimentos.
        pipeline = build_random_forest_pipeline()

        pipeline.fit(
            X_train,
            y_train
        )

        # ---------------------------------------------------------
        # 4. PREDICCIONES
        # ---------------------------------------------------------

        predictions = pipeline.predict(
            X_validation
        )

        # ---------------------------------------------------------
        # 5. CONSTRUCCIÓN DEL DATASET DE ERRORES
        # ---------------------------------------------------------

        # Creamos una copia de X_validation para analizar
        # posteriormente cada predicción.
        errors = X_validation.copy()

        # Conservamos el id original del registro.
        #
        # train_test_split mantiene los índices originales,
        # por lo que podemos recuperar el id desde el dataset.
        errors["id"] = dataset.loc[
            X_validation.index,
            "id"
        ]

        # Valor real registrado por ESED.
        errors["COSM2_REAL"] = y_validation

        # Valor estimado por Random Forest.
        errors["COSM2_PREDICTED"] = predictions

        # Error absoluto expresado en USD/m².
        errors["ABSOLUTE_ERROR"] = (
            errors["COSM2_REAL"]
            - errors["COSM2_PREDICTED"]
        ).abs()

        # ---------------------------------------------------------
        # 6. DISTRIBUCIÓN DE LOS ERRORES
        # ---------------------------------------------------------

        absolute_error = errors["ABSOLUTE_ERROR"]

        error_distribution = {

            "<=50": int(
                (absolute_error <= 50).sum()
            ),

            "50-100": int(
                (
                    (absolute_error > 50)
                    & (absolute_error <= 100)
                ).sum()
            ),

            "100-200": int(
                (
                    (absolute_error > 100)
                    & (absolute_error <= 200)
                ).sum()
            ),

            "200-500": int(
                (
                    (absolute_error > 200)
                    & (absolute_error <= 500)
                ).sum()
            ),

            ">500": int(
                (absolute_error > 500).sum()
            )
        }

        # ---------------------------------------------------------
        # 7. MAYORES ERRORES
        # ---------------------------------------------------------

        # Recuperamos los 20 registros donde el modelo
        # tuvo las mayores diferencias entre valor real
        # y valor estimado.
        largest_errors = (
            errors
            .sort_values(
                by="ABSOLUTE_ERROR",
                ascending=False
            )
            .head(20)
        )

        # ---------------------------------------------------------
        # 8. IMPORTANCIA DE VARIABLES
        # ---------------------------------------------------------

        # Utilizamos Permutation Importance.
        #
        # En lugar de confiar únicamente en la importancia
        # interna de los árboles, medimos qué ocurre cuando
        # alteramos una variable del conjunto de validación.
        #
        # Si al mezclar una variable el error aumenta mucho,
        # esa variable contiene información importante.
        importance_result = permutation_importance(

            estimator=pipeline,

            X=X_validation,

            y=y_validation,

            # Utilizamos MAE para medir el impacto.
            scoring="neg_mean_absolute_error",

            # Cada variable se mezcla 5 veces para obtener
            # una estimación más estable.
            n_repeats=5,

            # Reproducibilidad.
            random_state=42,

            # Utiliza los núcleos disponibles.
            n_jobs=-1
        )

        # ---------------------------------------------------------
        # 9. ORGANIZACIÓN DE IMPORTANCIAS
        # ---------------------------------------------------------

        feature_importance = []

        for index, feature in enumerate(X_validation.columns):

            feature_importance.append({
                "variable": feature,

                # Cuánto empeora aproximadamente el MAE
                # cuando la variable deja de aportar información.
                "importance": float(
                    importance_result.importances_mean[index]
                ),

                # Variabilidad entre las distintas repeticiones.
                "std": float(
                    importance_result.importances_std[index]
                )
            })

        # Ordenamos desde la variable más importante
        # hasta la que menos aporta.
        feature_importance.sort(
            key=lambda item: item["importance"],
            reverse=True
        )

        # ---------------------------------------------------------
        # 10. MÉTRICAS GENERALES
        # ---------------------------------------------------------

        mae = mean_absolute_error(
            y_validation,
            predictions
        )

        rmse = sqrt(
            mean_squared_error(
                y_validation,
                predictions
            )
        )

        r2 = r2_score(
            y_validation,
            predictions
        )

        # ---------------------------------------------------------
        # 11. RESPUESTA
        # ---------------------------------------------------------

        return {
            "model": "Random Forest Diagnostic",

            "validation_records": int(
                len(X_validation)
            ),

            "metrics": {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            },

            "error_percentiles": {
                "50": float(
                    absolute_error.quantile(0.50)
                ),
                "75": float(
                    absolute_error.quantile(0.75)
                ),
                "90": float(
                    absolute_error.quantile(0.90)
                ),
                "95": float(
                    absolute_error.quantile(0.95)
                ),
                "99": float(
                    absolute_error.quantile(0.99)
                )
            },

            "error_distribution": error_distribution,

            "feature_importance": feature_importance,

            "largest_errors": (
                largest_errors[
                    [
                        "id",
                        "COSM2_REAL",
                        "COSM2_PREDICTED",
                        "ABSOLUTE_ERROR",
                        "codcantf",
                        "CTIPOBR",
                        "NUPICAL",
                        "CSUTE",
                        "CARCO"
                    ]
                ]
                .to_dict(orient="records")
            )
        }