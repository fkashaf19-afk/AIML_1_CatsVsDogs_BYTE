"""
AIML_1_CatsVsDogs_BYTE
Standalone inference script.

Loads the saved model and predicts cat/dog on any image(s) you point it at.

Usage:
    python inference.py path/to/image.jpg
    python inference.py path/to/folder_of_images/
"""

import sys
import os
import numpy as np
import tensorflow as tf

IMG_SIZE = (160, 160)
CLASS_NAMES = ["cat", "dog"]
MODEL_PATH = os.path.join("outputs", "cats_vs_dogs_model.keras")


def load_and_prep(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return arr


def predict(model, image_paths):
    batch = np.stack([load_and_prep(p) for p in image_paths])
    probs = model.predict(batch, verbose=0).flatten()
    for path, prob in zip(image_paths, probs):
        label = CLASS_NAMES[int(prob >= 0.5)]
        confidence = prob if prob >= 0.5 else 1 - prob
        print(f"{path:50s} -> {label:5s} (confidence: {confidence:.3f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path_or_folder>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        valid_ext = (".jpg", ".jpeg", ".png")
        image_paths = [
            os.path.join(target, f)
            for f in sorted(os.listdir(target))
            if f.lower().endswith(valid_ext)
        ]
    else:
        image_paths = [target]

    if not image_paths:
        print("No valid images found.")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)

    predict(model, image_paths)
