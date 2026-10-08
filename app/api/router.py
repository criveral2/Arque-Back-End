from fastapi import APIRouter

from app.api.routes.dataset import router as dataset_router
from app.api.routes.health import router as health_router
from app.api.routes.model import router as model_router


router = APIRouter()

router.include_router(health_router)
router.include_router(dataset_router)
router.include_router(model_router)