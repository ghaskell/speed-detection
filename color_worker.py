#!/usr/bin/env python3
"""
Vehicle color worker — names each vehicle's color with CLIP (zero-shot) and
writes it to the database. Runs as a separate low-priority process so CLIP's
torch threads never compete with YOLO's in the main process.
"""

import os
import sqlite3
import time

import cv2
import numpy as np

CLIP_MODEL = "ViT-B-32"
CLIP_WEIGHTS = "laion2b_s34b_b79k"

# Prompt color -> stored color. Near-identical shades are merged because this
# camera can't reliably separate them (silver vs gray, red vs maroon).
COLOR_GROUPS = {
    "white": "white",
    "silver": "silver/gray",
    "gray": "silver/gray",
    "black": "black",
    "red": "red/maroon",
    "maroon": "red/maroon",
    "blue": "blue",
    "dark blue": "blue",
    "green": "green",
    "yellow": "yellow/gold",
    "gold": "yellow/gold",
    "orange": "orange",
    "brown": "brown/tan",
    "tan": "brown/tan",
    "beige": "brown/tan",
    "purple": "purple",
}

PROMPT_TEMPLATES = [
    "a photo of a {} car.",
    "a photo of a {} truck.",
    "a {} vehicle on a street.",
]


def load_model():
    import torch
    import open_clip

    torch.set_num_threads(1)
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_WEIGHTS)
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

    names = list(COLOR_GROUPS.keys())
    with torch.no_grad():
        text_features = []
        for name in names:
            feats = model.encode_text(tokenizer([t.format(name) for t in PROMPT_TEMPLATES]))
            feats /= feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0)
            text_features.append(mean / mean.norm())
        text_features = torch.stack(text_features)
    return {"model": model, "preprocess": preprocess, "text": text_features, "names": names}


def classify(bundle, bgr_crop):
    """Return (color, confidence 0-1) for a BGR vehicle crop."""
    import torch
    from PIL import Image

    image = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        feats = bundle["model"].encode_image(bundle["preprocess"](image).unsqueeze(0))
        feats /= feats.norm(dim=-1, keepdim=True)
        probs = (100 * feats @ bundle["text"].T).softmax(dim=-1)[0].numpy()

    totals = {}
    for name, p in zip(bundle["names"], probs):
        group = COLOR_GROUPS[name]
        totals[group] = totals.get(group, 0.0) + float(p)
    color = max(totals, key=totals.get)
    return color, round(totals[color], 2)


def run(queue, db_file):
    """Process entry point: read (row_id, jpeg_bytes) from queue, write color to vehicle_passes."""
    try:
        os.nice(10)
    except OSError:
        pass
    try:
        bundle = load_model()
    except Exception as e:
        print(f"[color] CLIP unavailable, vehicle colors disabled: {e}", flush=True)
        return
    print(f"[color] CLIP {CLIP_MODEL} loaded", flush=True)

    conn = sqlite3.connect(db_file, timeout=30)
    while True:
        row_id, jpeg = queue.get()
        try:
            crop = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            color, confidence = classify(bundle, crop)
            conn.execute("UPDATE vehicle_passes SET color = ?, color_confidence = ? WHERE id = ?",
                         (color, confidence, row_id))
            conn.commit()
        except Exception as e:
            print(f"[color] row {row_id} failed: {e}", flush=True)
            time.sleep(0.1)
