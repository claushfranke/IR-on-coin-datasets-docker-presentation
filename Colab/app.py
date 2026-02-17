"""
🪙 Münz-Bildsuche – Museum Touchscreen App
Nutzt VGG16-Modelle für Typ- und Prägestätten-Erkennung antiker Münzen.
Extrahiert die Logik aus dem IR-for-types-and-mints Notebook.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import cv2
import ast
import math
import requests
import zipfile
from io import BytesIO
from PIL import Image as PILImage
import matplotlib.cm as cm
import skimage.transform

# ─────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
LAST_CONV_LAYER = "block5_conv3"
TEMP_DIR = "temp_museum"
GALLERY_COLS = 4  # 2 Reihen à 4 = 8 Münzen

st.set_page_config(
    page_title="Münz-Bilderkennung",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Münz-Daten: 3 Basis-Münzen mit echten CN-URLs
# ─────────────────────────────────────────────
COINS_BASE = [
    {
        "type_id": "513",
        "name": "CN Typ 513",
        "mint": "Maroneia",
        "region": "Thrakien",
        "dating": "ca. 500–480 v. Chr.",
        "nominal": "Drachme, AR",
        "obv_url": "https://data.corpus-nummorum.eu/storage/coins/3454/img/3979/o/thumbnails/lg.jpeg",
        "rev_url": "https://data.corpus-nummorum.eu/storage/coins/3454/img/3979/r/thumbnails/lg.jpeg",
        "obv_desc": "Protome eines springenden Pferdes nach links.",
        "rev_desc": "Sternblume im quadratum incusum.",
        "link": "https://www.corpus-nummorum.eu/types/513",
    },
    {
        "type_id": "8862",
        "name": "CN Typ 8862",
        "mint": "Pergamon",
        "region": "Mysien",
        "dating": "ca. 241–197 v. Chr.",
        "nominal": "Tetradrachme, AR",
        "obv_url": "https://data.corpus-nummorum.eu/storage/coins/32574/img/35626/o/thumbnails/lg.jpeg",
        "rev_url": "https://data.corpus-nummorum.eu/storage/coins/32574/img/35626/r/thumbnails/lg.jpeg",
        "obv_desc": "Kopf des Philetairos mit Binde (taenia) umwundenen Efeukranz nach rechts.",
        "rev_desc": "Athena nach links thronend, in der vorgestreckten rechten Hand ein Kranz haltend, der linke Arm ruht auf Schild.",
        "link": "https://www.corpus-nummorum.eu/types/8862",
    },
    {
        "type_id": "513",
        "name": "CN Münze 1775 (Typ 513)",
        "mint": "Maroneia",
        "region": "Thrakien",
        "dating": "ca. 500–480 v. Chr.",
        "nominal": "Drachme, AR, 15 mm, 3.50 g",
        "obv_url": "https://data.corpus-nummorum.eu/storage/coins/1775/img/14001/o/thumbnails/lg.jpeg",
        "rev_url": "https://data.corpus-nummorum.eu/storage/coins/1775/img/14001/r/thumbnails/lg.jpeg",
        "obv_desc": "Protome eines springenden Pferdes nach links. Perlkreis.",
        "rev_desc": "Sternblume, bestehend aus vier Hauptstrahlen und drei kleinen Strahlen im quadratum incusum.",
        "link": "https://www.corpus-nummorum.eu/coins/1775",
    },
]

# Auf 8 Einträge erweitern durch Wiederholung
COINS = []
for i, coin in enumerate((COINS_BASE * 3)[:8]):
    entry = dict(coin)
    entry["display_idx"] = i
    COINS.append(entry)

# ─────────────────────────────────────────────
# Custom CSS – Touch-optimiert
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(160deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .stButton > button {
        width: 100%;
        min-height: 70px;
        font-size: 22px !important;
        font-weight: 600;
        border-radius: 16px;
        border: none;
        margin: 8px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        background: linear-gradient(135deg, #e2b04a 0%, #c5943a 100%);
        color: #1a1a2e !important;
    }
    .stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(226, 176, 74, 0.5);
    }
    .stButton > button:active { transform: scale(0.98); }
    .coin-card {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .coin-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    }
    h1, h2, h3 { color: #e2b04a !important; font-family: 'Georgia', serif; }
    h1 {
        text-align: center; font-size: 3rem !important;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5); padding: 1rem 0 0.5rem;
    }
    .subtitle {
        text-align: center; color: rgba(255,255,255,0.7);
        font-size: 1.3rem; margin-bottom: 2rem;
    }
    .desc-label { color: #e2b04a; font-weight: 700; font-size: 0.95rem; margin-bottom: 2px; }
    .desc-text { color: rgba(255,255,255,0.85); font-size: 1.05rem; margin-bottom: 0.5rem; line-height: 1.4; }
    .coin-meta {
        color: rgba(255,255,255,0.5); font-size: 0.85rem;
        text-align: center; margin-top: 0.3rem; line-height: 1.5;
    }
    .coin-meta strong { color: rgba(255,255,255,0.7); }
    .result-box {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(226,176,74,0.3);
        border-radius: 16px; padding: 1.2rem; margin: 0.8rem 0;
    }
    .result-rank { color: #e2b04a; font-size: 1.6rem; font-weight: 700; }
    .result-label { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
    .result-value { color: white; font-size: 1.15rem; font-weight: 600; }
    hr { border-color: rgba(226,176,74,0.3); }
    [data-testid="stSidebar"] { display: none; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Bild-Hilfsfunktionen
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def download_image(url: str):
    """Lädt ein Bild von einer URL und gibt ein PIL-Image zurück."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return PILImage.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        st.error(f"Bild konnte nicht geladen werden: {e}")
        return None


def save_pil_image(img, path: str):
    img.save(path, "JPEG")


def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(img.shape[0] for img in img_list)
    resized = [
        cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation=interpolation)
        for img in img_list
    ]
    return cv2.hconcat(resized)


def bound_image_dim(image, min_size=None, max_size=None):
    dtype = image.dtype
    (height, width, *_) = image.shape
    scale = 1
    if min_size is not None:
        scale = max(1, min_size / min(height, width))
    if max_size is not None:
        if round(max(height, width) * scale) > max_size:
            scale = max_size / max(height, width)
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(height * scale), round(width * scale)),
            order=1, mode="constant", preserve_range=True,
        )
    return image.astype(dtype), max(height, width)


def square_pad_image(image, size):
    (height, width, *_) = image.shape
    pad_h = (size - height) / 2
    pad_w = (size - width) / 2
    return np.pad(
        image,
        ((math.floor(pad_h), math.ceil(pad_h)),
         (math.floor(pad_w), math.ceil(pad_w)),
         (0, 0)),
        mode="constant",
    )


def combine_images(obv_path: str, rev_path: str) -> str:
    img_obv = cv2.imread(obv_path)
    img_rev = cv2.imread(rev_path)
    if img_obv.shape[0] != img_rev.shape[0]:
        combined = hconcat_resize([img_obv, img_rev])
    else:
        combined = cv2.hconcat([img_obv, img_rev])
    h, w, *_ = combined.shape
    mx, mn = max(h, w), min(h, w)
    padded, mx = bound_image_dim(combined, mn, mx)
    padded = square_pad_image(padded, mx)
    out_path = os.path.join(TEMP_DIR, "combined.jpg")
    cv2.imwrite(out_path, padded)
    return out_path


# ─────────────────────────────────────────────
# Modell-Funktionen
# ─────────────────────────────────────────────

def ensure_models_downloaded():
    needed = ["vgg16_types_95.keras", "vgg16_mints_96.keras", "dict_types_95.txt", "dict_mints_96.txt"]
    if all(os.path.exists(f) for f in needed):
        return True

    st.info("⬇️ Modelle werden heruntergeladen… Dies geschieht nur beim ersten Start.")
    progress = st.progress(0, text="Lade Modelle…")

    try:
        if not os.path.exists("vgg16_types_95.keras"):
            progress.progress(10, text="Lade Typ-Modell…")
            _download_gdrive("1m0N1pqY1mF50_XykAZsvk8lJkyXfKlhw", "vgg16_types.zip")
            with zipfile.ZipFile("vgg16_types.zip", "r") as z:
                z.extractall(".")
            os.remove("vgg16_types.zip")

        if not os.path.exists("vgg16_mints_96.keras"):
            progress.progress(40, text="Lade Prägestätten-Modell…")
            _download_gdrive("1LRhWPSIgWGcmrCOo2nZzpYfo6q7NC91e", "vgg16_mints.zip")
            with zipfile.ZipFile("vgg16_mints.zip", "r") as z:
                z.extractall(".")
            os.remove("vgg16_mints.zip")

        if not os.path.exists("dict_types_95.txt"):
            progress.progress(70, text="Lade Typ-Wörterbuch…")
            _download_gdrive("1lJFaMwvTde_oxxF2FK5jKP4BaYf1Tgzi", "dict_types_95.txt")

        if not os.path.exists("dict_mints_96.txt"):
            progress.progress(80, text="Lade Prägestätten-Wörterbuch…")
            _download_gdrive("1DwaiC5Ec_-rxhaJVzAyyhEX9yujfW3JM", "dict_mints_96.txt")

        progress.progress(100, text="Modelle bereit!")
        progress.empty()
        return True
    except Exception as e:
        st.error(f"❌ Fehler beim Herunterladen: {e}")
        return False


def _download_gdrive(file_id: str, dest: str):
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


@st.cache_resource
def load_models():
    tf.get_logger().setLevel("ERROR")
    model_types = tf.keras.models.load_model("vgg16_types_95.keras")
    model_mints = tf.keras.models.load_model("vgg16_mints_96.keras")
    with open("dict_types_95.txt") as f:
        dict_types = ast.literal_eval(f.read())
    with open("dict_mints_96.txt") as f:
        dict_mints = ast.literal_eval(f.read())
    return model_types, model_mints, dict_types, dict_mints


# ─────────────────────────────────────────────
# Vorhersage-Funktionen (aus Notebook adaptiert)
# ─────────────────────────────────────────────

def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(array, axis=0)


def top_5(preds):
    indices = np.argsort(preds[0])[::-1][:5]
    top = [(i, preds[0][i]) for i in indices if preds[0][i] > 0]
    return [t[0] for t in top], [t[1] for t in top]


def translate(top5, classes, dict_types, dict_mints):
    d = dict_types if classes == "types" else dict_mints
    return [d[k] for k in top5]


def make_gradcam_heatmap(img_array, model, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def create_gradcam_overlay(img_path, heatmap):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed = jet_heatmap * 0.3 + img
    return keras.preprocessing.image.array_to_img(superimposed)


def run_prediction(coin, mode="types"):
    """Führt Typ- oder Prägestätten-Vorhersage aus."""
    model_types, model_mints, dict_types, dict_mints = load_models()
    model = model_types if mode == "types" else model_mints

    os.makedirs(TEMP_DIR, exist_ok=True)
    obv_path = os.path.join(TEMP_DIR, "obv.jpg")
    rev_path = os.path.join(TEMP_DIR, "rev.jpg")

    obv_img = download_image(coin["obv_url"])
    rev_img = download_image(coin["rev_url"])
    if obv_img is None or rev_img is None:
        return None, None, None

    save_pil_image(obv_img, obv_path)
    save_pil_image(rev_img, rev_path)

    combined_path = combine_images(obv_path, rev_path)

    preprocess = keras.applications.resnet.preprocess_input
    img_array = preprocess(get_img_array(combined_path, size=IMAGE_SIZE))
    preds = model.predict(img_array, verbose=0)
    indices, probs = top_5(preds)
    labels = translate(indices, mode, dict_types, dict_mints)

    model.layers[-1].activation = None
    heatmap = make_gradcam_heatmap(img_array, model)
    gradcam_img = create_gradcam_overlay(combined_path, heatmap)

    combined_img = PILImage.open(combined_path)
    return list(zip(labels, probs)), gradcam_img, combined_img


# ─────────────────────────────────────────────
# UI: Galerie-Ansicht
# ─────────────────────────────────────────────

def render_gallery():
    st.markdown("# 🪙 Corpus Nummorum – Münzerkennung")
    st.markdown(
        '<p class="subtitle">Tippen Sie auf eine Münze, um den Typ oder die Prägestätte bestimmen zu lassen</p>',
        unsafe_allow_html=True,
    )

    for row_start in range(0, len(COINS), GALLERY_COLS):
        cols = st.columns(GALLERY_COLS, gap="large")
        for col_idx, col in enumerate(cols):
            idx = row_start + col_idx
            if idx >= len(COINS):
                break
            coin = COINS[idx]

            with col:
                st.markdown('<div class="coin-card">', unsafe_allow_html=True)

                obv_img = download_image(coin["obv_url"])
                rev_img = download_image(coin["rev_url"])

                c1, c2 = st.columns(2)
                with c1:
                    if obv_img:
                        st.image(obv_img, use_container_width=True)
                    st.markdown(
                        f'<div class="desc-label">Vorderseite</div>'
                        f'<div class="desc-text">{coin["obv_desc"]}</div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    if rev_img:
                        st.image(rev_img, use_container_width=True)
                    st.markdown(
                        f'<div class="desc-label">Rückseite</div>'
                        f'<div class="desc-text">{coin["rev_desc"]}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div class="coin-meta">'
                    f'<strong>{coin["name"]}</strong><br>'
                    f'{coin["mint"]} · {coin["region"]}<br>'
                    f'{coin["dating"]} · {coin["nominal"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if st.button("🔍  Münze analysieren", key=f"coin_{idx}", use_container_width=True):
                    st.session_state.mode = "results"
                    st.session_state.selected_coin = coin
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UI: Ergebnis-Ansicht
# ─────────────────────────────────────────────

def render_results():
    coin = st.session_state.selected_coin

    st.markdown("# 🔍 Analyseergebnis")

    col_back, _, _ = st.columns([1, 2, 1])
    with col_back:
        if st.button("⬅️  Zurück zur Galerie", key="back", use_container_width=True):
            st.session_state.mode = "gallery"
            st.session_state.selected_coin = None
            st.session_state.pop("prediction_results", None)
            st.rerun()

    st.markdown("---")

    # Ausgewählte Münze groß anzeigen
    st.markdown("### 📌 Ausgewählte Münze")
    _, col_obv, col_rev, _ = st.columns([0.5, 2, 2, 0.5])

    with col_obv:
        obv = download_image(coin["obv_url"])
        if obv:
            st.image(obv, use_container_width=True)
        st.markdown(
            f'<div class="desc-label">Vorderseite (Avers)</div>'
            f'<div class="desc-text">{coin["obv_desc"]}</div>',
            unsafe_allow_html=True,
        )

    with col_rev:
        rev = download_image(coin["rev_url"])
        if rev:
            st.image(rev, use_container_width=True)
        st.markdown(
            f'<div class="desc-label">Rückseite (Revers)</div>'
            f'<div class="desc-text">{coin["rev_desc"]}</div>',
            unsafe_allow_html=True,
        )

    # Münz-Metadaten
    st.markdown(
        f'<div style="text-align:center; margin: 1rem 0;">'
        f'<span class="desc-label" style="font-size:1.1rem">{coin["name"]}</span><br>'
        f'<span class="desc-text">{coin["mint"]} · {coin["region"]} · {coin["dating"]} · {coin["nominal"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Analyse-Buttons
    col1, col2 = st.columns(2)
    with col1:
        run_types = st.button("🏛️  Typ bestimmen", key="pred_type", use_container_width=True)
    with col2:
        run_mints = st.button("📍  Prägestätte bestimmen", key="pred_mint", use_container_width=True)

    if run_types or run_mints:
        mode = "types" if run_types else "mints"
        mode_label = "Typ" if mode == "types" else "Prägestätte"
        with st.spinner(f"🔄 {mode_label}-Analyse läuft…"):
            results, gradcam, combined = run_prediction(coin, mode=mode)
        if results is None:
            st.error("❌ Analyse fehlgeschlagen.")
            return
        st.session_state.prediction_results = {
            "results": results, "gradcam": gradcam,
            "combined": combined, "mode": mode,
        }

    # Ergebnisse persistent anzeigen
    if "prediction_results" in st.session_state:
        pred = st.session_state.prediction_results
        mode = pred["mode"]
        results = pred["results"]

        mode_label = "Typen" if mode == "types" else "Prägestätten"
        st.markdown(f"### 🎯 Erkannte {mode_label}")

        for rank, (label, prob) in enumerate(results, 1):
            prob_pct = prob * 100 if prob < 1 else prob
            medal = ["🥇", "🥈", "🥉", "4.", "5."][rank - 1] if rank <= 5 else f"{rank}."

            with st.container():
                c_rank, c_info, c_link = st.columns([0.8, 3, 1.5])
                with c_rank:
                    st.markdown(
                        f'<div class="result-box" style="text-align:center">'
                        f'<span class="result-rank">{medal}</span></div>',
                        unsafe_allow_html=True,
                    )
                with c_info:
                    info_label = "Typ-ID" if mode == "types" else "Prägestätte"
                    info_value = f"Typ {label}" if mode == "types" else label
                    st.markdown(
                        f'<div class="result-box">'
                        f'<div class="result-label">{info_label}</div>'
                        f'<div class="result-value">{info_value}</div>'
                        f'<div class="result-label" style="margin-top:0.5rem">Wahrscheinlichkeit</div>'
                        f'<div class="result-value">{prob_pct:.1f} %</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with c_link:
                    if mode == "types":
                        link = f"https://www.corpus-nummorum.eu/types/{label}"
                    else:
                        link = f"https://www.corpus-nummorum.eu/search/types?type=quicksearch&q={label.replace(' ', '+')}"
                    st.markdown(
                        f'<div class="result-box" style="text-align:center; padding-top:1.5rem">'
                        f'<a href="{link}" target="_blank" '
                        f'style="color:#e2b04a; font-size:1.1rem; text-decoration:none;">'
                        f'🔗 Auf CN ansehen</a></div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("---")

        # GradCAM
        st.markdown("### 🔬 Modell-Aufmerksamkeit (GradCAM)")
        st.markdown(
            '<div class="desc-text" style="max-width:800px">'
            "Die Heatmap zeigt, welche Bildbereiche für die Vorhersage besonders "
            "wichtig waren. Warme Farben (gelb/rot) = hohe Aufmerksamkeit.</div>",
            unsafe_allow_html=True,
        )
        col_comb, col_grad = st.columns(2)
        with col_comb:
            st.markdown(
                '<div class="desc-label" style="text-align:center; font-size:1.1rem">'
                "Kombiniertes Eingabebild</div>", unsafe_allow_html=True,
            )
            if pred["combined"]:
                st.image(pred["combined"], use_container_width=True)
        with col_grad:
            st.markdown(
                '<div class="desc-label" style="text-align:center; font-size:1.1rem">'
                "GradCAM Heatmap</div>", unsafe_allow_html=True,
            )
            if pred["gradcam"]:
                st.image(pred["gradcam"], use_container_width=True)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    if "mode" not in st.session_state:
        st.session_state.mode = "gallery"
    if "selected_coin" not in st.session_state:
        st.session_state.selected_coin = None

    os.makedirs(TEMP_DIR, exist_ok=True)

    if not ensure_models_downloaded():
        st.stop()

    if st.session_state.mode == "gallery":
        render_gallery()
    elif st.session_state.mode == "results":
        if st.session_state.selected_coin:
            render_results()
        else:
            st.session_state.mode = "gallery"
            st.rerun()


if __name__ == "__main__":
    main()
