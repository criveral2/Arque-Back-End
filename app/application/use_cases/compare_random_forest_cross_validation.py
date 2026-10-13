import numpy as np

from sklearn.model_selection import (
    KFold,
    cross_validate,
    train_test_split
)

from app.domain.model_features import (
    CATEGORICAL_VARIABLES,
    NUMERIC_VARIABLES,
    TARGET_VARIABLE
)

from app.infrastructure.ml.pipelines import (
    build_random_forest_pipeline,
    build_random_forest_log_pipeline
)


class CompareRandomForestCrossValidationUseCase:

    def __init__(self, dataset_repository):

        # Repositorio que permite cargar
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

        X = dataset[
            predictor_variables
        ].copy()

        y = dataset[
            TARGET_VARIABLE
        ].copy()

        # ---------------------------------------------------------
        # 2. SEPARACIÓN TRAIN / VALIDATION
        # ---------------------------------------------------------

        # Conservamos exactamente la misma división 80/20
        # utilizada durante los experimentos anteriores.
        #
        # IMPORTANTE:
        # el conjunto de validación NO se utiliza en este análisis.
        (
            X_train,
            _,
            y_train,
            _
        ) = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        # ---------------------------------------------------------
        # 3. CONFIGURACIÓN DEL CROSS-VALIDATION
        # ---------------------------------------------------------

        # Dividimos los 18.216 registros de entrenamiento
        # en cinco subconjuntos.
        #
        # shuffle=True evita depender del orden original
        # de los registros.
        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        # Materializamos los folds para garantizar
        # que ambos modelos utilicen EXACTAMENTE
        # las mismas divisiones.
        folds = list(
            cv.split(
                X_train,
                y_train
            )
        )

        # ---------------------------------------------------------
        # 4. MÉTRICAS
        # ---------------------------------------------------------

        # cross_validate utiliza valores negativos para
        # métricas de error porque scikit-learn interpreta
        # valores mayores como mejores.
        #
        # Más adelante invertimos el signo.
        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        }

        # ---------------------------------------------------------
        # 5. RANDOM FOREST NORMAL
        # ---------------------------------------------------------

        random_forest = (
            build_random_forest_pipeline()
        )

        rf_results = cross_validate(
            random_forest,
            X_train,
            y_train,
            cv=folds,
            scoring=scoring,

            # Se utiliza un solo proceso externo porque
            # Random Forest ya utiliza paralelización interna.
            n_jobs=1
        )

        # ---------------------------------------------------------
        # 6. RANDOM FOREST + LOG
        # ---------------------------------------------------------

        random_forest_log = (
            build_random_forest_log_pipeline()
        )

        rf_log_results = cross_validate(
            random_forest_log,
            X_train,
            y_train,
            cv=folds,
            scoring=scoring,
            n_jobs=1
        )

        # ---------------------------------------------------------
        # 7. FORMATEAR RESULTADOS
        # ---------------------------------------------------------

        rf_mae = -rf_results["test_mae"]
        rf_rmse = -rf_results["test_rmse"]
        rf_r2 = rf_results["test_r2"]

        rf_log_mae = -rf_log_results[
            "test_mae"
        ]

        rf_log_rmse = -rf_log_results[
            "test_rmse"
        ]

        rf_log_r2 = rf_log_results[
            "test_r2"
        ]

        # ---------------------------------------------------------
        # 8. RESPUESTA
        # ---------------------------------------------------------

        return {
            "training_records": int(
                len(X_train)
            ),

            "cross_validation": {
                "folds": 5,
                "shuffle": True,
                "random_state": 42
            },

            "random_forest": {
                "MAE": {
                    "mean": float(
                        np.mean(rf_mae)
                    ),
                    "std": float(
                        np.std(rf_mae)
                    ),
                    "folds": [
                        float(value)
                        for value in rf_mae
                    ]
                },

                "RMSE": {
                    "mean": float(
                        np.mean(rf_rmse)
                    ),
                    "std": float(
                        np.std(rf_rmse)
                    ),
                    "folds": [
                        float(value)
                        for value in rf_rmse
                    ]
                },

                "R2": {
                    "mean": float(
                        np.mean(rf_r2)
                    ),
                    "std": float(
                        np.std(rf_r2)
                    ),
                    "folds": [
                        float(value)
                        for value in rf_r2
                    ]
                }
            },

            "random_forest_log": {
                "MAE": {
                    "mean": float(
                        np.mean(rf_log_mae)
                    ),
                    "std": float(
                        np.std(rf_log_mae)
                    ),
                    "folds": [
                        float(value)
                        for value in rf_log_mae
                    ]
                },

                "RMSE": {
                    "mean": float(
                        np.mean(rf_log_rmse)
                    ),
                    "std": float(
                        np.std(rf_log_rmse)
                    ),
                    "folds": [
                        float(value)
                        for value in rf_log_rmse
                    ]
                },

                "R2": {
                    "mean": float(
                        np.mean(rf_log_r2)
                    ),
                    "std": float(
                        np.std(rf_log_r2)
                    ),
                    "folds": [
                        float(value)
                        for value in rf_log_r2
                    ]
                }
            }
        }