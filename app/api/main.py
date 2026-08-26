"""FastAPI application — Sprint 11.

RUN LOCALLY:
    uvicorn app.api.main:app --reload --port 8000

DOCS:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from contextlib import asynccontextmanager

import logfire
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Observability FIRST
from src.observability.langsmith_setup import setup_langsmith
from src.observability.logfire_setup import setup_logfire

from app.api.routes import debate
from app.api.schemas.response import HealthResponse
from src.core.config import get_settings
from src.core.logger import get_logger

langsmith_ok = setup_langsmith()
logfire_ok   = setup_logfire()

logger   = get_logger(__name__)
settings = get_settings()


import os



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("DebateEdge API starting...")

    # Warm up model on startup so first request is fast
    try:
        from src.services.debate_service import DebateService
        svc = DebateService()
        logger.info("Model warm-up complete.")
    except Exception as e:
        logger.warning(f"Warm-up failed (model may not exist): {e}")

    yield

    logger.info("DebateEdge API shutting down.")


app = FastAPI(
    title="DebateEdge API",
    description=(
        "Production-grade AI Debate and Argument Coach. "
        "Powered by LangGraph, Groq, and LangSmith."
    ),
    version="0.1.0",
    lifespan=lifespan,
    contact={
        "name":  "Kafil Aslam",
        "url":   "https://github.com/Kafil-Aslam-13/DebateEdge",
        "email": "aslamkafil13@gmail.com",
    },
)

# Instrument FastAPI with Logfire AFTER app creation
if logfire_ok:
    logfire.instrument_fastapi(app)

ALLOWED_ORIGINS = [
    "http://localhost:3000", 
    "http://localhost:5173",         # local dev
    "http://localhost:8000",
]

_frontend_url = os.getenv("FRONTEND_URL", "")
if _frontend_url:
    ALLOWED_ORIGINS.append(_frontend_url)

extra = os.getenv("EXTRA_ORIGINS", "")
if extra:
    ALLOWED_ORIGINS.extend([o.strip() for o in extra.split(",") if o.strip()])

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # tighten to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(debate.router)


@app.get("/", tags=["Root"])
def root() -> dict:
    return {
        "name":    "DebateEdge API",
        "version": "0.1.0",
        "docs":    "/docs",
        "health":  "/api/v1/health",
        "github":  "https://github.com/Kafil-Aslam-13/DebateEdge",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.environment,
    )