"""
main.py
=======
FastAPI application entry point.
Run with:  uvicorn gui.backend.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .routers import training, inference, preprocessing, filesystem

app = FastAPI(
    title="SUPARCO Super-Resolution",
    description="SwinIR training, inference, and preprocessing.",
    version="1.0.0",
)

# CORS — allow the Vite dev server on port 5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(preprocessing.router)
app.include_router(filesystem.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "SUPARCO SR"}


# Serve the built React frontend in production
_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
