"""
AIML_1_CatsVsDogs_BYTE
Task 1 - Image Classification (Cats vs Dogs)

Trains a binary image classifier (cat vs dog) using transfer learning
on MobileNetV2, evaluates it on a held-out test split, and saves:
  - the trained model (SavedModel + .h5)
  - test-set metrics (accuracy, precision, recall, F1)
  - a confusion matrix image
  - a training history (accuracy/loss curves) image
  - 10 sample inference images with predicted vs ground-truth labels

Dataset is downloaded directly (no tensorflow_datasets dependency, which
avoids a known protobuf version conflict on some Colab/py3.13 environments).

Run:
    pip install -r requirements.txt
    python train.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS_HEAD = 6          # train just the classification head
EPOCHS_FINE_TUNE = 4     # then fine-tune the top of the base model
SEED = 42
OUTPUT_DIR = "outputs"
CLASS_NAMES = ["cat", "dog"]  # alphabetical: Cat < Dog

DATASET_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
# This is Microsoft's official hosted copy of the same Kaggle "Dogs vs. Cats"
# images (https://www.kaggle.com/c/dogs-vs-cats), used directly to avoid the
# tensorflow_datasets/tensorflow_metadata dependency chain.
import zipfile

EXTRACT_DIR = "cats_and_dogs_data"
if not os.path.exists(EXTRACT_DIR):
    zip_path = tf.keras.utils.get_file(
        "kagglecatsanddogs_5340.zip",
        origin=DATASET_URL,
        extract=False,
    )
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)

PET_IMAGES_DIR = os.path.join(EXTRACT_DIR, "PetImages")

print(f"Dataset source: Microsoft-hosted copy of the Kaggle 'Dogs vs. Cats' "
      f"dataset ({DATASET_URL})")

# The raw dataset contains a handful of corrupted / non-JFIF images that
# will crash the image decoder. Filter them out first (standard step for
# this dataset - see the official TF image classification tutorial).
print("Scanning for corrupted images...")
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(PET_IMAGES_DIR, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "rb") as fobj:
                is_jfif = b"JFIF" in fobj.peek(10)
        except Exception:
            is_jfif = False
        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)
print(f"Deleted {num_skipped} corrupted/invalid images.")

# 80% train, remaining 20% split evenly into val/test.
raw_train = tf.keras.utils.image_dataset_from_directory(
    PET_IMAGES_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)
raw_val_test = tf.keras.utils.image_dataset_from_directory(
    PET_IMAGES_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

val_test_batches = tf.data.experimental.cardinality(raw_val_test)
raw_test = raw_val_test.take(val_test_batches // 2)
raw_val = raw_val_test.skip(val_test_batches // 2)

print(f"Class names found on disk: {raw_train.class_names}")

AUTOTUNE = tf.data.AUTOTUNE


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    return image, label


train_ds = (
    raw_train.map(augment, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
val_ds = raw_val.prefetch(AUTOTUNE)
test_ds = raw_test.prefetch(AUTOTUNE)

# Keep an un-prefetched, un-batched-for-display copy of test set for later
# (sample inference grid needs raw images before caching/prefetch nuances).
test_ds_for_samples = raw_test

# ---------------------------------------------------------------------------
# 2. Model - MobileNetV2 transfer learning
# ---------------------------------------------------------------------------
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze for phase 1

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---------------------------------------------------------------------------
# 3. Phase 1: train the classification head
# ---------------------------------------------------------------------------
print("\n--- Phase 1: training classification head ---")
history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

# ---------------------------------------------------------------------------
# 4. Phase 2: fine-tune top layers of the base model
# ---------------------------------------------------------------------------
print("\n--- Phase 2: fine-tuning top of base model ---")
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE_TUNE)

# combine histories for plotting
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss = history1.history["loss"] + history2.history["loss"]
val_loss = history1.history["val_loss"] + history2.history["val_loss"]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="train acc")
plt.plot(val_acc, label="val acc")
plt.axvline(EPOCHS_HEAD - 1, color="gray", linestyle="--", label="fine-tune start")
plt.title("Accuracy")
plt.xlabel("epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label="train loss")
plt.plot(val_loss, label="val loss")
plt.axvline(EPOCHS_HEAD - 1, color="gray", linestyle="--", label="fine-tune start")
plt.title("Loss")
plt.xlabel("epoch")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=150)
plt.close()
print(f"Saved training curves to {OUTPUT_DIR}/training_history.png")

# ---------------------------------------------------------------------------
# 5. Evaluate on the held-out test set
# ---------------------------------------------------------------------------
print("\n--- Evaluating on test set ---")
y_true, y_pred = [], []
for images, labels in test_ds:
    probs = model.predict(images, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)
    y_true.extend(labels.numpy().flatten().astype(int).tolist())
    y_pred.extend(preds.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc_score = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary"
)

print(f"Test Accuracy : {acc_score:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-score      : {f1:.4f}")
print("\nFull classification report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Test Accuracy : {acc_score:.4f}\n")
    f.write(f"Precision     : {precision:.4f}\n")
    f.write(f"Recall        : {recall:.4f}\n")
    f.write(f"F1-score      : {f1:.4f}\n\n")
    f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - Cats vs Dogs (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print(f"Saved confusion matrix to {OUTPUT_DIR}/confusion_matrix.png")

# ---------------------------------------------------------------------------
# 6. Save 10 sample inference images (predicted vs ground truth)
# ---------------------------------------------------------------------------
print("\n--- Generating sample inference images ---")
sample_images, sample_labels = next(iter(test_ds_for_samples.unbatch().batch(10)))
sample_probs = model.predict(sample_images, verbose=0).flatten()
sample_preds = (sample_probs >= 0.5).astype(int)

plt.figure(figsize=(15, 7))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i].numpy().astype("uint8"))
    true_label = CLASS_NAMES[int(sample_labels[i][0])]
    pred_label = CLASS_NAMES[sample_preds[i]]
    confidence = sample_probs[i] if sample_preds[i] == 1 else 1 - sample_probs[i]
    color = "green" if true_label == pred_label else "red"
    plt.title(
        f"Pred: {pred_label} ({confidence:.2f})\nTrue: {true_label}",
        color=color,
        fontsize=10,
    )
    plt.axis("off")
plt.suptitle("Sample Inference Results (Test Set)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_inference.png"), dpi=150)
plt.close()
print(f"Saved sample inference grid to {OUTPUT_DIR}/sample_inference.png")

# ---------------------------------------------------------------------------
# 7. Save the trained model
# ---------------------------------------------------------------------------
model.save(os.path.join(OUTPUT_DIR, "cats_vs_dogs_model.keras"))
model.save(os.path.join(OUTPUT_DIR, "cats_vs_dogs_model.h5"))
print(f"\nSaved trained model to {OUTPUT_DIR}/cats_vs_dogs_model.keras (.h5 too)")

print("\nAll done. Deliverables are in the 'outputs/' folder:")
print("  - cats_vs_dogs_model.keras / .h5  (trained model)")
print("  - metrics.txt                     (accuracy, precision, recall, F1)")
print("  - confusion_matrix.png")
print("  - training_history.png")
print("  - sample_inference.png            (10 labeled sample predictions)")
