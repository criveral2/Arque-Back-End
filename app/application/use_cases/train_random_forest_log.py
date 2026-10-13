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


class TrainRandomForestLogUseCase:

    def __init__(
        self,
        dataset_repository
    ):
        # Repositorio que permite acceder
        # al dataset preparado de ESED 2023.
        self.dataset_repository = dataset_repository

    def execute(self):

        # ---------------------------------------------------------
        # 1. CARGA DEL DATASET
        # ---------------------------------------------------------

        dataset = self.dataset_repository.load()

        # ---------------------------------------------------------
        # 2. SELECCIÓN DE PREDICTORES Y OBJETIVO
        # ---------------------------------------------------------

        predictor_variables = (
            CATEGORICAL_VARIABLES
            + NUMERIC_VARIABLES
        )

        # Variables utilizadas para realizar la predicción.
        X = dataset[
            predictor_variables
        ].copy()

        # Variable que deseamos estimar.
        y = dataset[
            TARGET_VARIABLE
        ].copy()

        # ---------------------------------------------------------
        # 3. DIVISIÓN TRAIN / VALIDATION
        # ---------------------------------------------------------

        # Utilizamos exactamente la misma división 80/20
        # y random_state=42 de los experimentos anteriores.
        #
        # Esto permite comparar los modelos de forma justa.
        (
            X_train,
            X_validation,
            y_train,
            y_validation
        ) = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        # ---------------------------------------------------------
        # 4. CONSTRUCCIÓN DEL MODELO
        # ---------------------------------------------------------

        # Random Forest cuya variable objetivo se transforma
        # internamente mediante log1p(COSM2).
        model = build_random_forest_log_pipeline()

        # ---------------------------------------------------------
        # 5. ENTRENAMIENTO
        # ---------------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ---------------------------------------------------------
        # 6. PREDICCIÓN
        # ---------------------------------------------------------

        # TransformedTargetRegressor aplica automáticamente
        # expm1() y devuelve COSM2 nuevamente en USD/m².
        predictions = model.predict(
            X_validation
        )

        # ---------------------------------------------------------
        # 7. EVALUACIÓN
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
        # 8. RESULTADO
        # ---------------------------------------------------------

        return {
            "model": "Random Forest Log Target",

            "training_records": int(
                len(X_train)
            ),

            "validation_records": int(
                len(X_validation)
            ),

            "parameters": {
                "n_estimators": 200,
                "random_state": 42,
                "target_transformation": "log1p"
            },

            "metrics": {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            }
        }