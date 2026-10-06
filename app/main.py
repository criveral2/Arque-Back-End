from fastapi import FastAPI

from app.api.router import router


app = FastAPI(
    title="Arqué API",
    description="API para la estimación de costos de construcción mediante aprendizaje automático",
    version="0.1.0"
)

app.include_router(
    router,
    prefix="/api/v1"
)