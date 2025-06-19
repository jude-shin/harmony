"""
src/api/server.py
Flask micro‑service for card identification.

Environment
-----------
MODEL_DIR   Absolute path to directory that contains a single `model.keras`
LABEL_FILE  Absolute path to CSV/JSON file used by `label_to_json`
PORT        Port to serve on (default 5000)
THRESHOLD   Default confidence threshold (0.6 if unset)

Author: you
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image

from config.constants import GAMES
from config.paths import MODELS_PATH
from utils.data_conversion import label_to_json

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
GAME = os.getenv("GAME", GAMES.LORCANA.value)  # e.g. "mtg", "pokemon"
MODEL_DIR = Path(os.getenv("MODEL_DIR", MODELS_PATH / GAME))
MODEL_PATH = MODEL_DIR / "model.keras"
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", 0.60))
PORT = int(os.getenv("PORT", 5000))

IMG_HEIGHT = 224  # update if your model expects something else
IMG_WIDTH = 224

# --------------------------------------------------------------------------- #
# App & model initialisation
# --------------------------------------------------------------------------- #
app = Flask(__name__)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except (IOError, ValueError) as exc:
    raise SystemExit(f"Could not load model at {MODEL_PATH}: {exc}") from exc


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize → scale to [0,1] → add batch dimension."""
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/healthz")
def health() -> tuple[str, int]:
    return "ok", 200


@app.post("/predict")
def predict() -> tuple[dict, int]:
    if "image" not in request.files:
        return jsonify(error="image file missing"), 400

    threshold = float(request.form.get("threshold", DEFAULT_THRESHOLD))

    # Load & preprocess ------------------------------------------------------ #
    try:
        pil_image = Image.open(request.files["image"].stream).convert("RGB")
        batch = preprocess_image(pil_image)
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"invalid image: {exc}"), 400

    # Inference -------------------------------------------------------------- #
    probs = model.predict(batch, verbose=False)[0]  # shape: (num_labels,)
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    # Payload ---------------------------------------------------------------- #
    label_json = json.loads(label_to_json(idx))  # <- pass the index, not full vector
    payload = {
        "prediction": label_json,
        "confidence": confidence,
        "threshold_exceeded": confidence >= threshold,
    }
    return jsonify(payload), 200


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Use `gunicorn -k gthread` in production; the builtin server is single‑threaded
    app.run(host="0.0.0.0", port=PORT)

