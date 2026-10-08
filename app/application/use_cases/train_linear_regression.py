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
    build_linear_regression_pipeline
)


class TrainLinearRegressionUseCase:

    def __init__(self, dataset_repository):

        # Repositorio que permitirá cargar
        # esed_2023_prepared.csv.
        self.dataset_repository = dataset_repository

    def execute(self):

        # ---------------------------------------------------------
        # 1. PREPARACIÓN DE DATOS
        # ---------------------------------------------------------

        # Reutilizamos el caso de uso que ya construimos
        # para separar:
        #
        # X_train
        # X_validation
        # y_train
        # y_validation
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

        # Construimos el Pipeline:
        #
        # preprocesamiento
        #       ↓
        # Regresión Lineal
        pipeline = build_linear_regression_pipeline()

        # ---------------------------------------------------------
        # 3. ENTRENAMIENTO
        # ---------------------------------------------------------

        # fit() realiza dos cosas automáticamente:
        #
        # 1. Ajusta el preprocesamiento utilizando SOLO X_train.
        # 2. Entrena la Regresión Lineal utilizando y_train.
        #
        # Esto evita que los datos de validación participen
        # accidentalmente durante el entrenamiento.
        pipeline.fit(
            X_train,
            y_train
        )

        # ---------------------------------------------------------
        # 4. PREDICCIONES
        # ---------------------------------------------------------

        # Utilizamos el conjunto de validación,
        # que el modelo nunca vio durante el entrenamiento.
        predictions = pipeline.predict(
            X_validation
        )

        # ---------------------------------------------------------
        # 5. MÉTRICAS DE EVALUACIÓN
        # ---------------------------------------------------------

        # MAE:
        # error absoluto promedio expresado en USD/m².
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
        # raíz cuadrada del MSE.
        #
        # También queda expresado en USD/m².
        rmse = sqrt(mse)

        # R²:
        # indica qué proporción de la variabilidad
        # de COSM2 es explicada por el modelo.
        r2 = r2_score(
            y_validation,
            predictions
        )

        # ---------------------------------------------------------
        # 6. RESULTADO
        # ---------------------------------------------------------

        return {
            "model": "Linear Regression",

            "training_records": int(len(X_train)),
            "validation_records": int(len(X_validation)),

            "metrics": {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            }
        }