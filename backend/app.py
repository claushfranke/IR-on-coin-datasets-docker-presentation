"""
🪙 Münz-Analyse Backend – FastAPI
REST-API zur Typ- und Prägestätten-Erkennung antiker Münzen.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from download_models import ensure_models
from models import models_available, load_models, run_analysis

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Lifespan: Modelle beim Start laden
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Beim Start Modelle herunterladen und laden."""
    logger.info("🪙 Münz-Analyse Backend startet...")

    # Modelle herunterladen falls nötig
    if not ensure_models():
        logger.error("Modelle konnten nicht geladen werden!")
        raise RuntimeError("Modell-Download fehlgeschlagen")

    # Modelle in den Speicher laden
    logger.info("Lade Modelle in den Speicher...")
    load_models()
    logger.info("✓ Backend bereit!")

    yield

    logger.info("Backend wird heruntergefahren.")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Münz-Analyse API",
    description="REST-API für die Typ- und Prägestätten-Erkennung antiker Münzen mittels VGG16.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS für Frontend-Zugriff
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request/Response Schemata
# ─────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    coin_id: str = Field(
        ...,
        description="ID der Münze, z.B. 'coin_01'",
        examples=["coin_01"],
    )
    mode: str = Field(
        ...,
        description="Analyse-Modus: 'types' (Typ) oder 'mints' (Prägestätte)",
        examples=["types", "mints"],
    )


class TypeImageSet(BaseModel):
    obverse_url: str
    reverse_url: str


class PredictionItem(BaseModel):
    rank: int
    label: str
    confidence: float
    cn_link: str
    display_label: str
    type_images: list[TypeImageSet] | None = None
    typology_texts: dict | None = None


class AnalyzeResponse(BaseModel):
    predictions: list[PredictionItem]
    gradcam_image: str = Field(description="Base64-encoded PNG der GradCAM-Heatmap")
    combined_image: str = Field(description="Base64-encoded PNG des kombinierten Eingabebilds")
    mode: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    message: str


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health-Check Endpoint."""
    loaded = models_available()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        models_loaded=loaded,
        message="Backend bereit" if loaded else "Modelle nicht verfügbar",
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_coin(request: AnalyzeRequest):
    """
    Analysiert eine Münze nach Typ oder Prägestätte.

    - **coin_id**: ID der Münze (z.B. "coin_01")
    - **mode**: "types" für Typ-Erkennung, "mints" für Prägestätten-Erkennung

    Gibt die Top-5 Vorhersagen mit Konfidenzwerten,
    eine GradCAM-Heatmap und das kombinierte Eingabebild zurück.
    """
    # Validierung
    if request.mode not in ("types", "mints"):
        raise HTTPException(
            status_code=400,
            detail=f"Ungültiger Modus: '{request.mode}'. Erlaubt: 'types', 'mints'",
        )

    if not models_available():
        raise HTTPException(
            status_code=503,
            detail="Modelle noch nicht geladen. Bitte warten.",
        )

    try:
        logger.info(f"Analyse gestartet: coin={request.coin_id}, mode={request.mode}")
        result = run_analysis(request.coin_id, request.mode)
        logger.info(
            f"Analyse abgeschlossen: coin={request.coin_id}, "
            f"top1={result['predictions'][0]['display_label']} "
            f"({result['predictions'][0]['confidence']:.1f}%)"
        )
        return AnalyzeResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Analyse-Fehler: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysefehler: {str(e)}")
