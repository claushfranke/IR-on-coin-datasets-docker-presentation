"""
Bildverarbeitungs-Hilfsfunktionen für die Münz-Analyse.
Extrahiert aus dem Colab-Notebook / Streamlit-App.
"""

import os
import math
import numpy as np
import cv2
import skimage.transform
from PIL import Image as PILImage
from io import BytesIO
import base64

import tensorflow as tf
from tensorflow import keras
import matplotlib.cm as cm


# ─────────────────────────────────────────────
# Konstanten
# ─────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
LAST_CONV_LAYER = "block5_conv3"
TEMP_DIR = "/tmp/coin_analysis"


# ─────────────────────────────────────────────
# Bild-Hilfsfunktionen
# ─────────────────────────────────────────────

def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    """Horizontale Verkettung mit Höhenanpassung."""
    h_min = min(img.shape[0] for img in img_list)
    resized = [
        cv2.resize(
            img,
            (int(img.shape[1] * h_min / img.shape[0]), h_min),
            interpolation=interpolation,
        )
        for img in img_list
    ]
    return cv2.hconcat(resized)


def bound_image_dim(image, min_size=None, max_size=None):
    """Bild auf min/max-Größe skalieren."""
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
            image,
            (round(height * scale), round(width * scale)),
            order=1,
            mode="constant",
            preserve_range=True,
        )
    return image.astype(dtype), max(height, width)


def square_pad_image(image, size):
    """Quadratisches Padding eines Bildes."""
    (height, width, *_) = image.shape
    pad_h = (size - height) / 2
    pad_w = (size - width) / 2
    return np.pad(
        image,
        (
            (math.floor(pad_h), math.ceil(pad_h)),
            (math.floor(pad_w), math.ceil(pad_w)),
            (0, 0),
        ),
        mode="constant",
    )


def combine_images(obv_path: str, rev_path: str) -> str:
    """
    Kombiniert Avers- und Revers-Bild zu einem quadratisch gepaddeten Bild.
    Gibt den Pfad zum kombinierten Bild zurück.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)

    img_obv = cv2.imread(obv_path)
    img_rev = cv2.imread(rev_path)

    if img_obv is None or img_rev is None:
        raise ValueError(f"Konnte Bilder nicht laden: obv={obv_path}, rev={rev_path}")

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
# GradCAM-Funktionen
# ─────────────────────────────────────────────

def get_img_array(img_path: str, size: tuple) -> np.ndarray:
    """Lädt ein Bild und bereitet es als Numpy-Array für das Modell vor."""
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(array, axis=0)


def make_gradcam_heatmap(img_array: np.ndarray, model, pred_index=None) -> np.ndarray:
    """Erzeugt eine GradCAM-Heatmap für die Modell-Vorhersage."""
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


def create_gradcam_overlay(img_path: str, heatmap: np.ndarray) -> PILImage.Image:
    """Überlagert die GradCAM-Heatmap auf das Originalbild."""
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


# ─────────────────────────────────────────────
# Encoding-Hilfsfunktionen
# ─────────────────────────────────────────────

def pil_to_base64(img: PILImage.Image, format: str = "PNG") -> str:
    """Konvertiert ein PIL-Image in einen Base64-String."""
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
