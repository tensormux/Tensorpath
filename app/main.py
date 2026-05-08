from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env from repo root before importing modules that may read env vars at
# import time (e.g. the agentic Forge orchestrator needs ANTHROPIC_API_KEY).
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

from app.api.forge_routes import router as forge_router  # noqa: E402
from app.api.routes import router as api_router  # noqa: E402
from app.ui.routes import router as ui_router  # noqa: E402

app = FastAPI(
    title="NeevPath",
    description="Inference optimization control plane for NeevCloud",
    version="0.1.0",
)

app.include_router(api_router)
app.include_router(forge_router)
app.include_router(ui_router)
