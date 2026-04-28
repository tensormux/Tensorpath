from fastapi import FastAPI

from app.api.routes import router as api_router
from app.ui.routes import router as ui_router

app = FastAPI(
    title="NeevPath",
    description="Inference optimization control plane for NeevCloud",
    version="0.1.0",
)

app.include_router(api_router)
app.include_router(ui_router)
