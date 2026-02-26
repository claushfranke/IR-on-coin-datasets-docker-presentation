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

_TYPOLOGY_TEXT_FIELDS = (
    "de_topography", "de_research", "de_typology", "de_metrology",
    "de_chronology", "de_special", "de_classic", "de_hellenistic", "de_imperial",
)

_typology_data: dict = {}  # normalisierter Name → {"url": str, "texts": dict, "nomisma_concated": str}
_mint_coords_cache: dict = {}  # nomisma-ID → {"lat": float, "lon": float, "region_de": str|None}


def fetch_typology_data() -> dict:
    """
    Lädt alle Münzstätten-Typologien von der CN-API und erstellt
    ein Lookup-Dictionary mit URL und deutschen Texten.
    Ergebnis wird gecacht.
    """
    global _typology_data
    if _typology_data:
        return _typology_data

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
            texts = {
                k: entry[k] for k in _TYPOLOGY_TEXT_FIELDS
                if entry.get(k)
            }
            record = {"url": url, "texts": texts, "nomisma_concated": entry.get("nomisma_concated")}
            for label_key in ("de_label", "en_label"):
                label_val = entry.get(label_key)
                if label_val:
                    _typology_data[label_val.strip().lower()] = record

        logger.info(f"Typologie-Daten geladen: {len(_typology_data)} Einträge")
    except Exception as e:
        logger.warning(f"Typologie-API Abruf fehlgeschlagen: {e}")

    return _typology_data


def fetch_all_mints() -> dict:
    """
    Lädt alle Münzstätten von der CN-API und erstellt ein
    Lookup-Dictionary nach normalisiertem Namen mit Koordinaten.
    Ergebnis wird gecacht.
    """
    global _mint_coords_cache
    if _mint_coords_cache:
        return _mint_coords_cache

    try:
        resp = requests.get(
            "https://data.corpus-nummorum.eu/api/mints",
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning(f"Mints-API Fehler: HTTP {resp.status_code}")
            return {}

        data = resp.json()
        for entry in data.get("contents", []):
            lat = entry.get("latitude")
            lon = entry.get("longitude")
            if not (lat and lon):
                continue
            try:
                coords = {
                    "lat": float(lat),
                    "lon": float(lon),
                    "region_de": entry.get("region_de"),
                }
            except (ValueError, TypeError):
                continue

            # Mehrere Namens-Keys eintragen für besseres Matching
            for key in ("name", "name_de", "name_en"):
                val = entry.get(key)
                if val:
                    _mint_coords_cache[val.strip().lower()] = coords

        logger.info(f"Münzstätten-Koordinaten geladen: {len(_mint_coords_cache)} Einträge")
    except Exception as e:
        logger.warning(f"Mints-API Abruf fehlgeschlagen: {e}")

    return _mint_coords_cache


def fetch_mint_coordinates(label) -> dict | None:
    """
    Gibt Koordinaten für eine Münzstätte zurück.
    Sucht direkt im Mints-API-Cache nach dem Label-Namen.
    Fallback: Verknüpfung über Typologie-Daten und nomisma-ID.
    """
    mint_coords = fetch_all_mints()
    key = str(label).strip().lower()

    # Direktes Namens-Matching
    if key in mint_coords:
        return mint_coords[key]

    # Fallback: über Typologie-Daten mit nomisma_concated
    tdata = fetch_typology_data()
    entry = tdata.get(key)
    if entry:
        nomisma = entry.get("nomisma_concated")
        if nomisma:
            # Suche nach nomisma-ID in den gecachten Mint-Namen
            # (Re-scan nötig, da Cache nach Namen nicht nach nomisma indiziert ist)
            try:
                resp = requests.get(
                    "https://data.corpus-nummorum.eu/api/mints",
                    timeout=15,
                )
                if resp.status_code == 200:
                    for m in resp.json().get("contents", []):
                        if m.get("nomisma") == nomisma:
                            lat, lon = m.get("latitude"), m.get("longitude")
                            if lat and lon:
                                return {
                                    "lat": float(lat),
                                    "lon": float(lon),
                                    "region_de": m.get("region_de"),
                                }
            except Exception:
                pass

    return None


def fetch_mint_typology_texts(label) -> dict | None:
    """Gibt die deutschen Typologietexte für eine Münzstätte zurück."""
    data = fetch_typology_data()
    entry = data.get(str(label).strip().lower())
    return entry["texts"] if entry else None


def build_cn_link(label, mode: str) -> str:
    """Erstellt einen Link zu Corpus Nummorum für das gegebene Label."""
    if mode == "types":
        return f"https://www.corpus-nummorum.eu/types/{label}"
    else:
        # Suche Typologie-Seite für die Münzstätte
        tdata = fetch_typology_data()
        entry = tdata.get(str(label).strip().lower())
        if entry:
            return entry["url"]
        # Fallback: Suche über Quicksearch
        query = str(label).replace(" ", "+")
        return f"https://www.corpus-nummorum.eu/search/types?type=quicksearch&q={query}"


def _ml(obj) -> str | None:
    """Extrahiert einen deutschen (oder englischen) Namen aus einem mehrsprachigen Feld."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj or None
    if isinstance(obj, dict):
        # Direkt name/de/en-Felder
        for key in ("de", "en"):
            if obj.get(key):
                return obj[key]
        # Verschachteltes name-Objekt
        name = obj.get("name")
        if isinstance(name, dict):
            return name.get("de") or name.get("en")
        if isinstance(name, str):
            return name
    return None


def fetch_type_data(type_id) -> dict:
    """
    Holt Beispielbilder UND Metadaten eines Typs von der CN-API in einem Aufruf.
    Gibt {"images": [...], "info": {...}} zurück.
    Bilder und Info werden unabhängig voneinander behandelt, damit ein Fehler
    bei den Metadaten nicht die Bilder-Anzeige kaputt macht.
    """
    result = {"images": [], "info": None}
    try:
        resp = requests.get(
            f"https://data.corpus-nummorum.eu/api/types/{type_id}",
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"CN-API Fehler für Typ {type_id}: HTTP {resp.status_code}")
            return result

        data = resp.json()
        contents = data.get("contents", [])
        if not contents:
            return result

        c = contents[0]

        # ── Bilder (eigener try/except) ──────────────────────────────
        try:
            images = []
            for img_set in (c.get("images") or [])[:3]:
                obv = img_set.get("obverse") or {}
                rev = img_set.get("reverse") or {}
                obv_thumb = (obv.get("thumbnail") or {}).get("lg") or obv.get("link")
                rev_thumb = (rev.get("thumbnail") or {}).get("lg") or rev.get("link")
                if obv_thumb and rev_thumb:
                    images.append({"obverse_url": obv_thumb, "reverse_url": rev_thumb})
            result["images"] = images
        except Exception as e:
            logger.warning(f"Bilder-Extraktion fehlgeschlagen für Typ {type_id}: {e}")

        # ── Metadaten (eigener try/except) ───────────────────────────
        try:
            # Münzstätte  →  mint.text.{de|en}
            mint_obj   = c.get("mint") or {}
            mint_name  = _ml(mint_obj.get("text"))

            # Region  →  mint.region.text.{de|en}
            region_obj  = (mint_obj.get("region") or {})
            region_name = _ml(region_obj.get("text"))

            # Datierung  →  date.text.{de|en}
            date_obj  = c.get("date") or {}
            date_text = _ml(date_obj.get("text"))

            # Epoche / Periode  →  date.period.text.{de|en}
            period_obj  = (date_obj.get("period") or {})
            period_name = _ml(period_obj.get("text"))

            # Denomination  →  denomination.text.{de|en}
            den_obj  = c.get("denomination") or {}
            den_name = _ml(den_obj.get("text"))

            # Material  →  material.text.{de|en}
            mat_obj  = c.get("material") or {}
            mat_name = _ml(mat_obj.get("text"))

            # Vorderseite  →  obverse.design.text.{de|en}  /  obverse.legend.string
            obv_obj    = c.get("obverse") or {}
            obv_design = (obv_obj.get("design") or {})
            obv_desc   = _ml(obv_design.get("text"))
            obv_legend_obj = obv_obj.get("legend") or {}
            obv_legend = obv_legend_obj.get("string") if isinstance(obv_legend_obj, dict) else None

            # Rückseite  →  reverse.design.text.{de|en}  /  reverse.legend.string
            rev_obj    = c.get("reverse") or {}
            rev_design = (rev_obj.get("design") or {})
            rev_desc   = _ml(rev_design.get("text"))
            rev_legend_obj = rev_obj.get("legend") or {}
            rev_legend = rev_legend_obj.get("string") if isinstance(rev_legend_obj, dict) else None

            # Metrologie  →  diameter.value_max  /  weight.value
            diam_obj = c.get("diameter") or {}
            wgt_obj  = c.get("weight")   or {}
            diameter = diam_obj.get("value_max") or diam_obj.get("value_min") or diam_obj.get("value")
            weight   = wgt_obj.get("value")

            info = {
                "mint":           mint_name,
                "region":         region_name,
                "date":           date_text,
                "period":         period_name,
                "denomination":   den_name,
                "material":       mat_name,
                "obverse_desc":   obv_desc,
                "obverse_legend": obv_legend,
                "reverse_desc":   rev_desc,
                "reverse_legend": rev_legend,
                "diameter_mm":    float(diameter) if diameter is not None else None,
                "weight_g":       float(weight)   if weight   is not None else None,
            }
            if any(v is not None for v in info.values()):
                result["info"] = info
        except Exception as e:
            logger.warning(f"Metadaten-Extraktion fehlgeschlagen für Typ {type_id}: {e}")

        return result

    except Exception as e:
        logger.warning(f"CN-API Abruf fehlgeschlagen für Typ {type_id}: {e}")
        return result


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

        # Bei Typ-Vorhersagen: Beispielbilder + Metadaten des Typs von der CN-API laden
        if mode == "types":
            type_data = fetch_type_data(label)
            pred_item["type_images"] = type_data["images"]
            pred_item["type_info"]   = type_data["info"]

        # Bei Münzstätten-Vorhersagen: Typologietexte und Koordinaten laden
        if mode == "mints":
            pred_item["typology_texts"] = fetch_mint_typology_texts(label)
            pred_item["mint_coordinates"] = fetch_mint_coordinates(label)

        predictions.append(pred_item)

    return {
        "predictions": predictions,
        "gradcam_image": pil_to_base64(gradcam_img),
        "combined_image": pil_to_base64(combined_img),
        "mode": mode,
    }
