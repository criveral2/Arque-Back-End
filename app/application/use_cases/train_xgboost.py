from math import sqrt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from app.application.use_cases.prepare_training_data import (
    PrepareTrainingDataUseCase
)

from app.infrastructure.ml.pipelines import (
    build_xgboost_pipeline
)


class TrainXGBoostUseCase:

    def __init__(self, dataset_repository):

        # Repositorio encargado de cargar
        # ESED 2023 previamente preparado.
        self.dataset_repository = dataset_repository

    def execute(self):

        # ---------------------------------------------------------
        # 1. PREPARACIÓN DE DATOS
        # ---------------------------------------------------------

        # Reutilizamos exactamente la misma división
        # utilizada por los otros modelos.
        #
        # Gracias a random_state=42 tendremos:
        #
        # 18.216 registros de entrenamiento
        # 4.554 registros de validación
        training_data_use_case = PrepareTrainingDataUseCase(
            self.dataset_repository
        )

        training_data = training_data_use_case.execute()

        X_train = training_data["X_train"]
        X_validation = training_data["X_validation"]

        y_train = training_data["y_train"]
        y_validation = training_data["y_validation"]

        # ---------------------------------------------------------
        # 2. CREACIÓN DEL PIPELINE
        # ---------------------------------------------------------

        # Pipeline:
        #
        # preprocesamiento
        #       ↓
        # XGBoost
        pipeline = build_xgboost_pipeline()

        # ---------------------------------------------------------
        # 3. ENTRENAMIENTO
        # ---------------------------------------------------------

        # El preprocesador y XGBoost aprenden
        # únicamente del conjunto de entrenamiento.
        pipeline.fit(
            X_train,
            y_train
        )

        # ---------------------------------------------------------
        # 4. PREDICCIÓN
        # ---------------------------------------------------------

        # Realizamos predicciones sobre los datos
        # que el modelo no vio durante el entrenamiento.
        predictions = pipeline.predict(
            X_validation
        )

        # ---------------------------------------------------------
        # 5. MÉTRICAS
        # ---------------------------------------------------------

        # MAE:
        # error absoluto promedio en USD/m².
        mae = mean_absolute_error(
            y_validation,
            predictions
        )

        # Error cuadrático medio.
        mse = mean_squared_error(
            y_validation,
            predictions
        )

        # RMSE:
        # penaliza más los errores grandes.
        rmse = sqrt(mse)

        # R²:
        # proporción de variabilidad de COSM2
        # explicada por el modelo.
        r2 = r2_score(
            y_validation,
            predictions
        )

        # ---------------------------------------------------------
        # 6. RESULTADOS
        # ---------------------------------------------------------

        return {
            "model": "XGBoost",

            "training_records": int(len(X_train)),
            "validation_records": int(len(X_validation)),

            "parameters": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            },

            "metrics": {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            }
        }