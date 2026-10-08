from pathlib import Path

from fastapi import APIRouter

from app.application.use_cases.train_linear_regression import (
    TrainLinearRegressionUseCase
)

from app.infrastructure.repositories.esed_csv_repository import (
    EsedCsvRepository
)

from app.application.use_cases.train_random_forest import (
    TrainRandomForestUseCase
)

from app.application.use_cases.train_xgboost import (
    TrainXGBoostUseCase
)


router = APIRouter(
    prefix="/model",
    tags=["Model"]
)


# Ruta raíz del proyecto.
BASE_DIR = Path(__file__).resolve().parents[3]


# Dataset 2023 ya preparado.
TRAINING_DATASET_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "esed_2023_prepared.csv"
)


@router.post("/train/linear-regression")
def train_linear_regression() -> dict:

    # Repositorio utilizado para acceder
    # al dataset preparado de 2023.
    repository = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Caso de uso encargado de entrenar
    # y evaluar la Regresión Lineal.
    use_case = TrainLinearRegressionUseCase(
        repository
    )

    return use_case.execute()

@router.post("/train/random-forest")
def train_random_forest() -> dict:

    # Repositorio que carga ESED 2023 preparado.
    repository = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Caso de uso encargado del entrenamiento
    # y evaluación de Random Forest.
    use_case = TrainRandomForestUseCase(
        repository
    )

    return use_case.execute()

@router.post("/train/xgboost")
def train_xgboost() -> dict:

    # Acceso al dataset preparado de 2023.
    repository = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Caso de uso encargado de entrenar
    # y evaluar XGBoost.
    use_case = TrainXGBoostUseCase(
        repository
    )

    return use_case.execute()