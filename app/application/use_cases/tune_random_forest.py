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
    build_random_forest_pipeline
)


class TuneRandomForestUseCase:

    # Semilla fija para que la búsqueda sea reproducible.
    RANDOM_STATE = 42

    def __init__(self, dataset_repository):

        # Repositorio que carga ESED 2023 preparado.
        self.dataset_repository = dataset_repository

    def execute(self):

        # ---------------------------------------------------------
        # 1. PREPARACIÓN DE LOS DATOS
        # ---------------------------------------------------------

        # Utilizamos exactamente la misma división 80/20
        # empleada en los modelos base.
        training_data_use_case = PrepareTrainingDataUseCase(
            self.dataset_repository
        )

        training_data = training_data_use_case.execute()

        X_train = training_data["X_train"]
        X_validation = training_data["X_validation"]

        y_train = training_data["y_train"]
        y_validation = training_data["y_validation"]

        # ---------------------------------------------------------
        # 2. PIPELINE BASE
        # ---------------------------------------------------------

        # Construimos el Pipeline:
        #
        # preprocesamiento
        #       ↓
        # Random Forest
        pipeline = build_random_forest_pipeline()

        # RandomizedSearchCV ejecutará varios entrenamientos
        # en paralelo.
        #
        # Para evitar utilizar todos los núcleos dos veces
        # simultáneamente, dejamos que cada Random Forest
        # individual utilice un solo núcleo.
        pipeline.set_params(
            model__n_jobs=1
        )

        # ---------------------------------------------------------
        # 3. ESPACIO DE HIPERPARÁMETROS
        # ---------------------------------------------------------

        # RandomizedSearchCV probará distintas combinaciones
        # tomadas de estos valores.
        parameter_distribution = {

            # Cantidad de árboles.
            "model__n_estimators": [
                100,
                200,
                300,
                500
            ],

            # Profundidad máxima de los árboles.
            #
            # None significa que el árbol puede crecer
            # hasta que las reglas internas lo detengan.
            "model__max_depth": [
                None,
                10,
                20,
                30
            ],

            # Número mínimo de registros necesarios
            # para dividir un nodo.
            "model__min_samples_split": [
                2,
                5,
                10
            ],

            # Número mínimo de registros que debe contener
            # una hoja del árbol.
            "model__min_samples_leaf": [
                1,
                2,
                4
            ],

            # Cantidad de características consideradas
            # al buscar una división.
            "model__max_features": [
                "sqrt",
                "log2",
                1.0
            ]
        }

        # ---------------------------------------------------------
        # 4. BÚSQUEDA DE HIPERPARÁMETROS
        # ---------------------------------------------------------

        # RandomizedSearchCV no prueba absolutamente todas
        # las combinaciones posibles.
        #
        # Selecciona aleatoriamente n_iter combinaciones.
        search = RandomizedSearchCV(

            estimator=pipeline,

            param_distributions=parameter_distribution,

            # Probaremos 20 configuraciones diferentes.
            n_iter=20,

            # Validación cruzada de 5 particiones.
            cv=5,

            # Buscamos minimizar RMSE.
            #
            # Scikit-learn trabaja con "mayor es mejor",
            # por eso internamente utiliza el valor negativo.
            scoring="neg_root_mean_squared_error",

            # Reproducibilidad de la búsqueda.
            random_state=self.RANDOM_STATE,

            # Utiliza los núcleos disponibles para ejecutar
            # diferentes entrenamientos en paralelo.
            n_jobs=-1,

            # Muestra progreso en consola.
            verbose=1,

            # Al terminar, vuelve a entrenar automáticamente
            # la mejor configuración utilizando todo X_train.
            refit=True
        )

        # ---------------------------------------------------------
        # 5. ENTRENAMIENTO Y OPTIMIZACIÓN
        # ---------------------------------------------------------

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
        # 6. MEJOR MODELO ENCONTRADO
        # ---------------------------------------------------------

        # best_estimator_ contiene el Pipeline completo
        # utilizando la mejor configuración encontrada.
        best_pipeline = search.best_estimator_

        # ---------------------------------------------------------
        # 7. EVALUACIÓN SOBRE EL 20 % DE VALIDACIÓN
        # ---------------------------------------------------------

        # Este conjunto NO participó en RandomizedSearchCV.
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

        # RandomizedSearchCV devuelve negativo porque
        # internamente intenta maximizar la puntuación.
        best_cv_rmse = -search.best_score_

        # ---------------------------------------------------------
        # 8. RESULTADOS
        # ---------------------------------------------------------

        return {
            "model": "Random Forest Optimized",

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