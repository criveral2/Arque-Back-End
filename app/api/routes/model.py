from pathlib import Path

from fastapi import (
    APIRouter,
    HTTPException,
    Request
)

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

from app.application.use_cases.tune_random_forest import (
    TuneRandomForestUseCase
)

from app.application.use_cases.tune_xgboost import (
    TuneXGBoostUseCase
)

from app.application.use_cases.analyze_random_forest import (
    AnalyzeRandomForestUseCase
)

from app.application.use_cases.analyze_target_ambiguity import (
    AnalyzeTargetAmbiguityUseCase
)

from app.application.use_cases.train_random_forest_log import (
    TrainRandomForestLogUseCase
)

from app.application.use_cases.analyze_random_forest_log import (
    AnalyzeRandomForestLogUseCase
)

from app.application.use_cases.compare_random_forest_cross_validation import (
    CompareRandomForestCrossValidationUseCase
)

from app.application.use_cases.evaluate_temporal_model import (
    EvaluateTemporalModelUseCase
)

from app.application.use_cases.analyze_temporal_shift import (
    AnalyzeTemporalShiftUseCase
)

from app.application.use_cases.train_production_model import (
    TrainProductionModelUseCase
)

from app.infrastructure.repositories.file_model_repository import (
    FileModelRepository
)

from fastapi import (
    APIRouter,
    HTTPException,
    Request
)

from app.api.schemas.prediction import (
    ConstructionPredictionRequest
)

from app.application.use_cases.predict_construction_cost import (
    PredictConstructionCostUseCase
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

# Dataset 2025 reservado para la
# evaluación temporal final.
TEST_DATASET_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "esed_2025_prepared.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "model.joblib"
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

@router.post("/tune/random-forest")
def tune_random_forest() -> dict:

    # Cargamos el mismo dataset ESED 2023 preparado
    # utilizado por los modelos base.
    repository = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Ejecutamos la optimización de hiperparámetros.
    use_case = TuneRandomForestUseCase(
        repository
    )

    return use_case.execute()

@router.post("/tune/xgboost")
def tune_xgboost() -> dict:

    # Se utiliza exactamente el mismo ESED 2023
    # empleado en los experimentos anteriores.
    repository = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Caso de uso encargado de buscar
    # mejores hiperparámetros para XGBoost.
    use_case = TuneXGBoostUseCase(
        repository
    )

    return use_case.execute()

@router.get("/analyze/random-forest")
def analyze_random_forest() -> dict:

    # Utilizamos ESED 2023 preparado.
    repository = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Ejecutamos el diagnóstico del modelo.
    use_case = AnalyzeRandomForestUseCase(
        repository
    )

    return use_case.execute()

@router.get("/analyze/target-ambiguity")
def analyze_target_ambiguity() -> dict:

    # Utilizamos ESED 2023 preparado.
    repository = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Ejecutamos el análisis de ambigüedad.
    use_case = AnalyzeTargetAmbiguityUseCase(
        repository
    )

    return use_case.execute()

@router.post("/train/random-forest-log")
def train_random_forest_log() -> dict:

    # Utilizamos exactamente el mismo dataset preparado
    # empleado por los demás modelos.
    repository = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Caso de uso encargado de entrenar Random Forest
    # utilizando la transformación logarítmica de COSM2.
    use_case = TrainRandomForestLogUseCase(
        repository
    )

    return use_case.execute()

@router.get("/analyze/random-forest-log")
def analyze_random_forest_log() -> dict:

    # Utilizamos exactamente el mismo
    # dataset ESED 2023 preparado.
    repository = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Ejecutamos el diagnóstico del
    # Random Forest con objetivo logarítmico.
    use_case = AnalyzeRandomForestLogUseCase(
        repository
    )

    return use_case.execute()

@router.get("/compare/random-forest-cross-validation")
def compare_random_forest_cross_validation() -> dict:

    # Utilizamos ESED 2023 preparado.
    repository = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Comparamos RF normal y RF con transformación
    # logarítmica mediante validación cruzada.
    use_case = (
        CompareRandomForestCrossValidationUseCase(
            repository
        )
    )

    return use_case.execute()

@router.post("/evaluate/temporal/random-forest-log")
def evaluate_temporal_random_forest_log() -> dict:

    # Dataset utilizado para entrenar el modelo.
    training_repository = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Dataset temporal utilizado únicamente
    # para evaluar la capacidad de generalización.
    test_repository = EsedCsvRepository(
        TEST_DATASET_PATH
    )

    # Ejecutamos la evaluación 2023 -> 2025.
    use_case = EvaluateTemporalModelUseCase(
        training_repository,
        test_repository
    )

    return use_case.execute()

@router.get("/analyze/temporal-shift")
def analyze_temporal_shift() -> dict:

    # Dataset histórico utilizado para entrenamiento.
    training_repository = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Dataset temporal utilizado en la prueba final.
    test_repository = EsedCsvRepository(
        TEST_DATASET_PATH
    )

    # Analizamos las diferencias entre ambos años
    # sin volver a entrenar ni modificar el modelo.
    use_case = AnalyzeTemporalShiftUseCase(
        training_repository,
        test_repository
    )

    return use_case.execute()

@router.post("/train/production")
def train_production_model() -> dict:

    # Dataset preparado de 2023.
    repository_2023 = EsedCsvRepository(
        TRAINING_DATASET_PATH
    )

    # Dataset preparado de 2025.
    repository_2025 = EsedCsvRepository(
        TEST_DATASET_PATH
    )

    # Repositorio encargado de guardar
    # models/model.joblib.
    model_repository = FileModelRepository(
        MODEL_PATH
    )

    # Entrenamos el modelo operativo de Arqué
    # utilizando toda la información disponible.
    use_case = TrainProductionModelUseCase(
        repository_2023,
        repository_2025,
        model_repository
    )

    return use_case.execute()

@router.post("/predict")
def predict_construction_cost(
    prediction_request: ConstructionPredictionRequest,
    request: Request
) -> dict:

    # ---------------------------------------------------------
    # 1. OBTENER MODELO CARGADO EN MEMORIA
    # ---------------------------------------------------------

    # El modelo es cargado durante el inicio
    # de la aplicación mediante lifespan.
    model = request.app.state.model

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo de predicción "
                "no se encuentra disponible."
            )
        )

    # ---------------------------------------------------------
    # 2. CREAR CASO DE USO
    # ---------------------------------------------------------

    use_case = PredictConstructionCostUseCase(
        model
    )

    # ---------------------------------------------------------
    # 3. EJECUTAR PREDICCIÓN
    # ---------------------------------------------------------

    return use_case.execute(
        prediction_request.model_dump()
    )