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
    build_random_forest_pipeline
)


class TrainRandomForestUseCase:

    def __init__(self, dataset_repository):

        # Repositorio que carga el dataset
        # preparado correspondiente a ESED 2023.
        self.dataset_repository = dataset_repository

    def execute(self):

        # ---------------------------------------------------------
        # 1. PREPARACIÓN DE LOS DATOS
        # ---------------------------------------------------------

        # Reutilizamos exactamente la misma división
        # entrenamiento / validación utilizada por
        # Regresión Lineal.
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

        # Construimos el Pipeline compuesto por:
        #
        # preprocesamiento
        #       ↓
        # Random Forest
        pipeline = build_random_forest_pipeline()

        # ---------------------------------------------------------
        # 3. ENTRENAMIENTO
        # ---------------------------------------------------------

        # Random Forest aprenderá únicamente utilizando
        # el conjunto de entrenamiento.
        pipeline.fit(
            X_train,
            y_train
        )

        # ---------------------------------------------------------
        # 4. PREDICCIÓN
        # ---------------------------------------------------------

        # Realizamos predicciones sobre los registros
        # que el modelo no utilizó para aprender.
        predictions = pipeline.predict(
            X_validation
        )

        # ---------------------------------------------------------
        # 5. EVALUACIÓN
        # ---------------------------------------------------------

        # MAE:
        # error absoluto promedio en USD/m².
        mae = mean_absolute_error(
            y_validation,
            predictions
        )

        # MSE:
        # error cuadrático medio.
        mse = mean_squared_error(
            y_validation,
            predictions
        )

        # RMSE:
        # penaliza especialmente los errores grandes.
        rmse = sqrt(mse)

        # R²:
        # capacidad del modelo para explicar
        # la variabilidad observada en COSM2.
        r2 = r2_score(
            y_validation,
            predictions
        )

        # ---------------------------------------------------------
        # 6. RESULTADOS
        # ---------------------------------------------------------

        return {
            "model": "Random Forest",

            "training_records": int(len(X_train)),
            "validation_records": int(len(X_validation)),

            "parameters": {
                "n_estimators": 200,
                "random_state": 42
            },

            "metrics": {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            }
        }