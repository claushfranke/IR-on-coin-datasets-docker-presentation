"""
ML-Modell-Verwaltung und Vorhersage-Pipeline.
Lädt VGG16-Modelle für Typ- und Prägestätten-Erkennung.
"""

import os
import ast
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils import (
    IMAGE_SIZE,
    TEMP_DIR,
    combine_images,
    get_img_array,
    make_gradcam_heatmap,
    create_gradcam_overlay,
    pil_to_base64,
)
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Modell-Pfade
# ─────────────────────────────────────────────
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
IMAGES_DIR = os.environ.get("IMAGES_DIR", "/app/images")

MODEL_FILES = {
    "types": os.path.join(MODELS_DIR, "vgg16_types_95.keras"),
    "mints": os.path.join(MODELS_DIR, "vgg16_mints_96.keras"),
}
DICT_FILES = {
    "types": os.path.join(MODELS_DIR, "dict_types_95.txt"),
    "mints": os.path.join(MODELS_DIR, "dict_mints_96.txt"),
}

# ─────────────────────────────────────────────
# Globaler Modell-Cache
# ─────────────────────────────────────────────
_models = {}
_dicts = {}


def models_available() -> bool:
    """Prüft ob alle Modell-Dateien vorhanden sind."""
    all_files = list(MODEL_FILES.values()) + list(DICT_FILES.values())
    return all(os.path.exists(f) for f in all_files)


def load_models():
    """Lädt alle Modelle und Dictionaries in den Cache."""
    global _models, _dicts

    if _models and _dicts:
        return

    tf.get_logger().setLevel("ERROR")
    logger.info("Lade ML-Modelle...")

    for mode in ("types", "mints"):
        logger.info(f"  Lade {mode}-Modell: {MODEL_FILES[mode]}")
        _models[mode] = tf.keras.models.load_model(MODEL_FILES[mode])

        logger.info(f"  Lade {mode}-Dictionary: {DICT_FILES[mode]}")
        with open(DICT_FILES[mode]) as f:
            _dicts[mode] = ast.literal_eval(f.read())

    logger.info("Alle Modelle geladen.")


def get_model(mode: str):
    """Gibt das Modell für den gegebenen Modus zurück."""
    if mode not in _models:
        load_models()
    return _models[mode]


def get_dict(mode: str):
    """Gibt das Label-Dictionary für den gegebenen Modus zurück."""
    if mode not in _dicts:
        load_models()
    return _dicts[mode]


# ─────────────────────────────────────────────
# Vorhersage-Funktionen
# ─────────────────────────────────────────────

def top_5(preds: np.ndarray):
    """Gibt die Top-5-Indizes und Wahrscheinlichkeiten zurück."""
    indices = np.argsort(preds[0])[::-1][:5]
    top = [(i, float(preds[0][i])) for i in indices if preds[0][i] > 0]
    return [t[0] for t in top], [t[1] for t in top]


def translate(top5_indices: list, mode: str) -> list:
    """Übersetzt Modell-Indizes in menschenlesbare Labels."""
    d = get_dict(mode)
    return [d[k] for k in top5_indices]


def build_cn_link(label, mode: str) -> str:
    """Erstellt einen Link zu Corpus Nummorum für das gegebene Label."""
    if mode == "types":
        return f"https://www.corpus-nummorum.eu/types/{label}"
    else:
        query = str(label).replace(" ", "+")
        return f"https://www.corpus-nummorum.eu/search/types?type=quicksearch&q={query}"


def run_analysis(coin_id: str, mode: str) -> dict:
    """
    Führt die komplette Analyse für eine Münze durch.

    Args:
        coin_id: z.B. "coin_01"
        mode: "types" oder "mints"

    Returns:
        Dictionary mit predictions, gradcam_image, combined_image, mode
    """
    os.makedirs(TEMP_DIR, exist_ok=True)

    obv_path = os.path.join(IMAGES_DIR, coin_id, "obverse.jpg")
    rev_path = os.path.join(IMAGES_DIR, coin_id, "reverse.jpg")

    if not os.path.exists(obv_path) or not os.path.exists(rev_path):
        raise FileNotFoundError(
            f"Münzbilder nicht gefunden: {obv_path}, {rev_path}"
        )

    # Bilder kombinieren
    combined_path = combine_images(obv_path, rev_path)

    # Modell laden und Vorhersage
    model = get_model(mode)
    preprocess = keras.applications.resnet.preprocess_input
    img_array = preprocess(get_img_array(combined_path, size=IMAGE_SIZE))

    preds = model.predict(img_array, verbose=0)
    indices, probs = top_5(preds)
    labels = translate(indices, mode)

    # GradCAM erzeugen
    model.layers[-1].activation = None
    heatmap = make_gradcam_heatmap(img_array, model)
    gradcam_img = create_gradcam_overlay(combined_path, heatmap)

    # Kombiniertes Bild laden
    combined_img = PILImage.open(combined_path)

    # Ergebnisse formatieren
    predictions = []
    for rank, (label, prob) in enumerate(zip(labels, probs), 1):
        prob_pct = prob * 100 if prob < 1 else prob
        predictions.append({
            "rank": rank,
            "label": str(label),
            "confidence": round(float(prob_pct), 2),
            "cn_link": build_cn_link(label, mode),
            "display_label": f"Typ {label}" if mode == "types" else str(label),
        })

    return {
        "predictions": predictions,
        "gradcam_image": pil_to_base64(gradcam_img),
        "combined_image": pil_to_base64(combined_img),
        "mode": mode,
    }
