from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.application.use_cases.inspect_dataset import InspectDatasetUseCase
from app.infrastructure.repositories.esed_csv_repository import EsedCsvRepository
from app.application.use_cases.analyze_candidate_variables import (
    AnalyzeCandidateVariablesUseCase
)
from app.application.use_cases.analyze_target_variable import (
    AnalyzeTargetVariableUseCase
)
from app.application.use_cases.inspect_target_outliers import (
    InspectTargetOutliersUseCase
)
from app.application.use_cases.analyze_target_consistency import (
    AnalyzeTargetConsistencyUseCase
)
from app.application.use_cases.inspect_problematic_records import (
    InspectProblematicRecordsUseCase
)
from app.application.use_cases.prepare_dataset import (
    PrepareDatasetUseCase
)

from app.infrastructure.repositories.csv_prepared_dataset_repository import (
    CsvPreparedDatasetRepository
)

router = APIRouter(
    prefix="/dataset",
    tags=["Dataset"]
)

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "6. 2023_ESED_BDD_definitiva.csv"
)

DATASET_2023_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "6. 2023_ESED_BDD_definitiva.csv"
)

DATASET_2025_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "6. 2025_ESED_BDD.csv"
)

PREPARED_2023_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "esed_2023_prepared.csv"
)

PREPARED_2025_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "esed_2025_prepared.csv"
)


@router.get("/info")
def dataset_info():

    repository = EsedCsvRepository(DATASET_PATH)

    use_case = InspectDatasetUseCase(repository)

    return use_case.execute()

@router.get("/analysis")
def dataset_analysis():
    repository = EsedCsvRepository(DATASET_PATH)
    use_case = AnalyzeCandidateVariablesUseCase(repository)

    return use_case.execute()

@router.get("/target-analysis")
def target_analysis() -> dict:

    # Se crea el repositorio que accederá al CSV de ESED 2023.
    repository = EsedCsvRepository(DATASET_PATH)

    # Se crea el caso de uso encargado de analizar COSM2.
    use_case = AnalyzeTargetVariableUseCase(repository)

    # FastAPI convierte automáticamente el diccionario retornado
    # por el caso de uso en una respuesta JSON.
    return use_case.execute()

@router.get("/target-outliers")
def target_outliers() -> dict:

    # Acceso al archivo ESED 2023.
    repository = EsedCsvRepository(DATASET_PATH)

    # Caso de uso encargado de inspeccionar
    # los valores extremos de COSM2.
    use_case = InspectTargetOutliersUseCase(repository)

    return use_case.execute()

@router.get("/target-consistency")
def target_consistency() -> dict:

    # Repositorio de acceso a ESED 2023.
    repository = EsedCsvRepository(DATASET_PATH)

    # Caso de uso encargado de evaluar la coherencia
    # entre COSM2, CVAE y CARCO.
    use_case = AnalyzeTargetConsistencyUseCase(repository)

    return use_case.execute()

@router.get("/problematic-records")
def problematic_records() -> dict:

    # Acceso a ESED 2023.
    repository = EsedCsvRepository(DATASET_PATH)

    # Ejecutamos la inspección de registros problemáticos.
    use_case = InspectProblematicRecordsUseCase(repository)

    return use_case.execute()

@router.post("/prepare/{year}")
def prepare_dataset(year: int) -> dict:

    # ---------------------------------------------------------
    # SELECCIÓN DE RUTAS SEGÚN EL AÑO
    # ---------------------------------------------------------

    # Asociamos cada año con:
    # - el archivo original
    # - el archivo preparado que se generará
    datasets = {
        2023: {
            "source": DATASET_2023_PATH,
            "target": PREPARED_2023_PATH
        },
        2025: {
            "source": DATASET_2025_PATH,
            "target": PREPARED_2025_PATH
        }
    }

    # Validamos que el año solicitado esté soportado.
    if year not in datasets:
        raise HTTPException(
            status_code=400,
            detail=f"Año ESED no soportado: {year}"
        )

    paths = datasets[year]

    # Repositorio que lee la base original.
    source_repository = EsedCsvRepository(
        paths["source"]
    )

    # Repositorio que guarda el dataset preparado.
    target_repository = CsvPreparedDatasetRepository(
        paths["target"]
    )

    # Caso de uso encargado de aplicar
    # las mismas reglas de preparación.
    use_case = PrepareDatasetUseCase(
        dataset_repository=source_repository,
        prepared_dataset_repository=target_repository
    )

    # Ejecutamos el proceso.
    result = use_case.execute()

    # Agregamos el año al resultado
    # para facilitar la trazabilidad.
    result["year"] = year

    return result