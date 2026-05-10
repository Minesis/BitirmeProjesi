"""
app/api/app.py
---------------
FastAPI application factory.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

import config
from .routes import init_repo, router


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Retail Analytics API",
        description="Real-time retail shelf analytics — multi-camera dwell + demographics",
        version="2.0.0",
    )

    # Allow Streamlit dashboard to call the API from a different port
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.on_event("startup")
    def _startup():
        db_path = config.get("database.path", default="data/analytics.db")
        init_repo(db_path)
        logger.info(f"API started. Database: {db_path}")

    return app
