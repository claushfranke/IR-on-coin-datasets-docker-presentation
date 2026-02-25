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

import requests

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


# ─────────────────────────────────────────────
# Münzstätten-Typologie-Lookup
# ─────────────────────────────────────────────

_typology_map: dict = {}  # normalisierter Name → Typologie-URL


def fetch_typology_map() -> dict:
    """
    Lädt alle Münzstätten-Typologien von der CN-API und erstellt
    ein Lookup-Dictionary: normalisierter Münzstättenname → API-URL.
    Ergebnis wird gecacht.
    """
    global _typology_map
    if _typology_map:
        return _typology_map

    try:
        resp = requests.get(
            "https://data.corpus-nummorum.eu/api/typology",
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"Typologie-API Fehler: HTTP {resp.status_code}")
            return {}

        data = resp.json()
        for entry in data.get("contents", []):
            entry_id = entry.get("id")
            if not entry_id:
                continue
            url = f"https://www.corpus-nummorum.eu/resources/typology/{entry_id}"
            for label_key in ("de_label", "en_label"):
                label_val = entry.get(label_key)
                if label_val:
                    _typology_map[label_val.strip().lower()] = url

        logger.info(f"Typologie-Lookup geladen: {len(_typology_map)} Einträge")
    except Exception as e:
        logger.warning(f"Typologie-API Abruf fehlgeschlagen: {e}")

    return _typology_map


def build_cn_link(label, mode: str) -> str:
    """Erstellt einen Link zu Corpus Nummorum für das gegebene Label."""
    if mode == "types":
        return f"https://www.corpus-nummorum.eu/types/{label}"
    else:
        # Suche Typologie-Seite für die Münzstätte
        tmap = fetch_typology_map()
        typology_url = tmap.get(str(label).strip().lower())
        if typology_url:
            return typology_url
        # Fallback: Suche über Quicksearch
        query = str(label).replace(" ", "+")
        return f"https://www.corpus-nummorum.eu/search/types?type=quicksearch&q={query}"


def fetch_type_images(type_id) -> list:
    """
    Holt Beispielbilder eines Typs von der CN-API.
    Gibt eine Liste von Imagesets zurück: [{obverse_url, reverse_url}, ...]
    """
    try:
        resp = requests.get(
            f"https://data.corpus-nummorum.eu/api/types/{type_id}",
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"CN-API Fehler für Typ {type_id}: HTTP {resp.status_code}")
            return []

        data = resp.json()
        contents = data.get("contents", [])
        if not contents:
            return []

        images = contents[0].get("images", [])
        result = []
        for img_set in images[:3]:  # Maximal 3 Imagesets
            obv = img_set.get("obverse", {})
            rev = img_set.get("reverse", {})
            obv_thumb = obv.get("thumbnail", {}).get("lg") or obv.get("link")
            rev_thumb = rev.get("thumbnail", {}).get("lg") or rev.get("link")
            if obv_thumb and rev_thumb:
                result.append({
                    "obverse_url": obv_thumb,
                    "reverse_url": rev_thumb,
                })
        return result

    except Exception as e:
        logger.warning(f"CN-API Abruf fehlgeschlagen für Typ {type_id}: {e}")
        return []


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

    # Softmax-Aktivierung entfernen für saubere GradCAM-Gradienten
    # (wie im Original-Notebook: model.layers[-1].activation = None)
    model.layers[-1].activation = None

    # GradCAM erzeugen (top-1 Klasse)
    heatmap = make_gradcam_heatmap(img_array, model, pred_index=int(indices[0]))
    gradcam_img = create_gradcam_overlay(combined_path, heatmap)

    # Kombiniertes Bild laden
    combined_img = PILImage.open(combined_path)

    # Ergebnisse formatieren
    predictions = []
    for rank, (label, prob) in enumerate(zip(labels, probs), 1):
        # Softmax-Output ist immer in [0, 1] → immer mit 100 multiplizieren
        prob_pct = prob * 100
        pred_item = {
            "rank": rank,
            "label": str(label),
            "confidence": float(prob_pct),
            "cn_link": build_cn_link(label, mode),
            "display_label": f"Typ {label}" if mode == "types" else str(label),
        }

        # Bei Typ-Vorhersagen: Beispielbilder des Typs von der CN-API laden
        if mode == "types":
            type_images = fetch_type_images(label)
            pred_item["type_images"] = type_images

        predictions.append(pred_item)

    return {
        "predictions": predictions,
        "gradcam_image": pil_to_base64(gradcam_img),
        "combined_image": pil_to_base64(combined_img),
        "mode": mode,
    }
