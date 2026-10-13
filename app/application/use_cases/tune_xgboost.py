from math import sqrt
from time import perf_counter

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import RandomizedSearchCV

from app.application.use_cases.prepare_training_data import (
    PrepareTrainingDataUseCase
)

from app.infrastructure.ml.pipelines import (
    build_xgboost_pipeline
)


class TuneXGBoostUseCase:

    # Semilla fija para garantizar que la búsqueda
    # pueda ser reproducida posteriormente.
    RANDOM_STATE = 42

    def __init__(self, dataset_repository):

        # Repositorio que permitirá acceder
        # a ESED 2023 previamente preparado.
        self.dataset_repository = dataset_repository

    def execute(self):

        # ---------------------------------------------------------
        # 1. PREPARACIÓN DE DATOS
        # ---------------------------------------------------------

        # Utilizamos exactamente la misma división
        # entrenamiento / validación que se empleó
        # con los modelos anteriores.
        training_data_use_case = PrepareTrainingDataUseCase(
            self.dataset_repository
        )

        training_data = training_data_use_case.execute()

        X_train = training_data["X_train"]
        X_validation = training_data["X_validation"]

        y_train = training_data["y_train"]
        y_validation = training_data["y_validation"]

        # ---------------------------------------------------------
        # 2. PIPELINE BASE DE XGBOOST
        # ---------------------------------------------------------

        # El Pipeline contiene:
        #
        # preprocesamiento
        #       ↓
        # XGBoost
        pipeline = build_xgboost_pipeline()

        # RandomizedSearchCV ejecutará varios modelos
        # simultáneamente.
        #
        # Para evitar paralelismo doble, dejamos que
        # cada instancia individual de XGBoost utilice
        # un solo núcleo.
        pipeline.set_params(
            model__n_jobs=1
        )

        # ---------------------------------------------------------
        # 3. HIPERPARÁMETROS A EXPLORAR
        # ---------------------------------------------------------

        parameter_distribution = {

            # Cantidad de árboles construidos.
            "model__n_estimators": [
                100,
                200,
                300,
                500,
                800
            ],

            # Cuánto aporta cada nuevo árbol.
            #
            # Valores pequeños producen un aprendizaje
            # más gradual.
            "model__learning_rate": [
                0.01,
                0.03,
                0.05,
                0.1,
                0.2
            ],

            # Profundidad máxima de cada árbol.
            "model__max_depth": [
                3,
                4,
                6,
                8,
                10
            ],

            # Controla cuántos registros mínimos
            # necesita un nodo para seguir creciendo.
            "model__min_child_weight": [
                1,
                3,
                5,
                10
            ],

            # Porcentaje de registros utilizados
            # en la construcción de cada árbol.
            "model__subsample": [
                0.6,
                0.8,
                1.0
            ],

            # Porcentaje de características utilizado
            # durante la construcción de cada árbol.
            "model__colsample_bytree": [
                0.6,
                0.8,
                1.0
            ],

            # Regularización L1.
            #
            # Ayuda a controlar modelos demasiado complejos.
            "model__reg_alpha": [
                0,
                0.1,
                1
            ],

            # Regularización L2.
            "model__reg_lambda": [
                1,
                5,
                10
            ]
        }

        # ---------------------------------------------------------
        # 4. BÚSQUEDA ALEATORIA
        # ---------------------------------------------------------

        search = RandomizedSearchCV(

            # Pipeline completo a optimizar.
            estimator=pipeline,

            # Espacio de parámetros definido anteriormente.
            param_distributions=parameter_distribution,

            # Cantidad de combinaciones que se probarán.
            n_iter=20,

            # Validación cruzada de cinco particiones.
            cv=5,

            # Utilizamos RMSE como criterio principal
            # para seleccionar la mejor configuración.
            scoring="neg_root_mean_squared_error",

            # Permite reproducir la misma búsqueda.
            random_state=self.RANDOM_STATE,

            # Ejecuta diferentes combinaciones
            # utilizando los núcleos disponibles.
            n_jobs=-1,

            # Muestra información del progreso
            # en la consola.
            verbose=1,

            # Al finalizar vuelve a entrenar automáticamente
            # la mejor configuración sobre todo X_train.
            refit=True
        )

        # ---------------------------------------------------------
        # 5. OPTIMIZACIÓN
        # ---------------------------------------------------------

        # Guardamos el tiempo únicamente como información
        # complementaria del experimento.
        start_time = perf_counter()

        search.fit(
            X_train,
            y_train
        )

        elapsed_seconds = (
            perf_counter()
            - start_time
        )

        # ---------------------------------------------------------
        # 6. MEJOR PIPELINE ENCONTRADO
        # ---------------------------------------------------------

        best_pipeline = search.best_estimator_

        # ---------------------------------------------------------
        # 7. VALIDACIÓN EXTERNA
        # ---------------------------------------------------------

        # El 20 % reservado nunca participó en
        # RandomizedSearchCV.
        predictions = best_pipeline.predict(
            X_validation
        )

        mae = mean_absolute_error(
            y_validation,
            predictions
        )

        mse = mean_squared_error(
            y_validation,
            predictions
        )

        rmse = sqrt(mse)

        r2 = r2_score(
            y_validation,
            predictions
        )

        # RandomizedSearchCV almacena el valor negativo
        # porque scikit-learn intenta maximizar la puntuación.
        best_cv_rmse = -search.best_score_

        # ---------------------------------------------------------
        # 8. RESULTADOS
        # ---------------------------------------------------------

        return {
            "model": "XGBoost Optimized",

            "training_records": int(len(X_train)),
            "validation_records": int(len(X_validation)),

            "search": {
                "iterations": 20,
                "cross_validation_folds": 5,
                "selection_metric": "RMSE",
                "best_cv_rmse": float(best_cv_rmse),
                "elapsed_seconds": float(elapsed_seconds)
            },

            "best_parameters": search.best_params_,

            "validation_metrics": {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            }
        }