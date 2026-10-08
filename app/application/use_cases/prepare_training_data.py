from sklearn.model_selection import train_test_split

from app.infrastructure.ml.preprocessing import (
    CATEGORICAL_VARIABLES,
    NUMERIC_VARIABLES
)


class PrepareTrainingDataUseCase:

    # Variable que intentaremos predecir.
    TARGET_VARIABLE = "COSM2"

    # Semilla fija utilizada para que la división de datos
    # sea reproducible.
    #
    # Si ejecutamos nuevamente el experimento,
    # obtendremos exactamente la misma separación.
    RANDOM_STATE = 42

    # 20 % de ESED 2023 se utilizará como validación.
    VALIDATION_SIZE = 0.20

    def __init__(self, dataset_repository):

        # Repositorio encargado de cargar
        # esed_2023_prepared.csv.
        self.dataset_repository = dataset_repository

    def execute(self):

        # ---------------------------------------------------------
        # 1. CARGA DEL DATASET PREPARADO
        # ---------------------------------------------------------

        dataset = self.dataset_repository.load()

        # ---------------------------------------------------------
        # 2. DEFINICIÓN DE X
        # ---------------------------------------------------------

        # X contiene únicamente las variables que el modelo
        # podrá utilizar para realizar predicciones.
        predictor_variables = (
            CATEGORICAL_VARIABLES
            + NUMERIC_VARIABLES
        )

        X = dataset[predictor_variables].copy()

        # ---------------------------------------------------------
        # 3. DEFINICIÓN DE y
        # ---------------------------------------------------------

        # y contiene la respuesta que queremos aprender:
        # el costo estimado por metro cuadrado.
        y = dataset[self.TARGET_VARIABLE].copy()

        # ---------------------------------------------------------
        # 4. DIVISIÓN ENTRENAMIENTO / VALIDACIÓN
        # ---------------------------------------------------------

        # ESED 2023 se divide en:
        #
        # 80 % entrenamiento
        # 20 % validación
        #
        # La validación nos permitirá comparar los modelos
        # antes de utilizar ESED 2025.
        X_train, X_validation, y_train, y_validation = (
            train_test_split(
                X,
                y,
                test_size=self.VALIDATION_SIZE,
                random_state=self.RANDOM_STATE
            )
        )

        return {
            "X_train": X_train,
            "X_validation": X_validation,
            "y_train": y_train,
            "y_validation": y_validation
        }