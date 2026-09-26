# AIML_1_CatsVsDogs_BYTE

**AVIP 2026 — AI/ML Engineering Track — Task 1: Image Classification (Cats vs Dogs)**

A binary image classifier that distinguishes cats from dogs, built with transfer
learning on MobileNetV2 (Keras/TensorFlow).

## Dataset

- **Source:** Microsoft's officially hosted copy of the
  [Kaggle "Dogs vs. Cats" dataset](https://www.kaggle.com/c/dogs-vs-cats)
  (~25,000 labeled images, same data used in the official TensorFlow
  tutorials): `https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip`
- **Access:** No manual download needed — `train.py` fetches and extracts it
  automatically via `tf.keras.utils.get_file` on first run (~800MB, cached
  locally afterward). A small number of corrupted/non-JFIF images in the raw
  archive are automatically detected and removed before training.
- **Split:** 80% train / 10% validation / 10% test (created via
  `image_dataset_from_directory`'s `validation_split`, with the validation
  portion further split in half for val/test).
- Labels: `0 = cat`, `1 = dog`.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train + evaluate

```bash
python train.py
```

This will:
1. Download and split the dataset.
2. Train a classification head on top of a frozen, ImageNet-pretrained
   MobileNetV2 (6 epochs), then fine-tune the top 30 layers (4 epochs).
3. Evaluate on the held-out test set and print/save accuracy, precision,
   recall, and F1-score.
4. Save a confusion matrix image and a training-curves image.
5. Save a grid of 10 sample test images with predicted vs. ground-truth labels.
6. Save the trained model.

All outputs land in `outputs/`:

```
outputs/
├── cats_vs_dogs_model.keras   # trained model (Keras v3 format)
├── cats_vs_dogs_model.h5      # trained model (H5 format)
├── metrics.txt                # accuracy, precision, recall, F1, full report
├── confusion_matrix.png
├── training_history.png       # accuracy/loss curves
└── sample_inference.png       # 10 labeled sample predictions
```

Training takes roughly 15–30 minutes on a free Colab GPU, longer on CPU.

## Run inference on your own images

```bash
python inference.py path/to/image.jpg
python inference.py path/to/folder_of_images/
```

Prints the predicted label and confidence for each image.

## Model details

- **Backbone:** MobileNetV2, ImageNet-pretrained, `include_top=False`
- **Head:** GlobalAveragePooling2D → Dropout(0.2) → Dense(1, sigmoid)
- **Input size:** 160×160×3
- **Training strategy:** freeze-then-fine-tune (train head @ lr=1e-3, then
  unfreeze top 30 base layers @ lr=1e-5)
- **Loss:** binary cross-entropy
- **Augmentation:** random horizontal flip, random brightness

## Repo naming convention

Per B.Y.T.E AVIP guidelines: `AIML_1_CatsVsDogs_BYTE`
