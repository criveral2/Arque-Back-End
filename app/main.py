from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.router import router
from app.application.use_cases.initialize_model import InitializeModelUseCase
from app.infrastructure.repositories.file_model_repository import FileModelRepository


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):

    model_repository = FileModelRepository(
        model_path=MODEL_PATH
    )

    initialize_model = InitializeModelUseCase(
        model_repository=model_repository
    )

    app.state.model = initialize_model.execute()

    yield


app = FastAPI(
    title="Arqué API",
    description="API para la estimación de costos de construcción mediante aprendizaje automático",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(
    router,
    prefix="/api/v1"
)