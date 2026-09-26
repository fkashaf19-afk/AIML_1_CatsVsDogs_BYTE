# Project Summary (Task 1 — Image Classification: Cats vs Dogs)

This project builds a binary image classifier that distinguishes cats from
dogs using transfer learning on MobileNetV2, pretrained on ImageNet. The
dataset (~23,000 images, TensorFlow Datasets' `cats_vs_dogs`, mirroring the
original Kaggle "Dogs vs. Cats" competition set) was split 80/10/10 into
train, validation, and test sets.

Training happened in two phases. First, the MobileNetV2 base was frozen and
only a lightweight classification head (global average pooling, dropout,
sigmoid output) was trained for 6 epochs at a learning rate of 1e-3. Then the
top 30 layers of the base model were unfrozen and fine-tuned for 4 more
epochs at a much lower learning rate (1e-5), which lets the network adapt its
higher-level features to cats and dogs specifically without destroying the
pretrained low-level features. Basic augmentation (random horizontal flips,
random brightness) reduced overfitting given the moderate dataset size.

On the held-out test split, the model was evaluated with accuracy,
precision, recall, and F1-score, plus a full confusion matrix, giving a
clear picture of how often cats are mistaken for dogs and vice versa. A
grid of 10 sample test images was also generated, each labeled with its
predicted class, confidence score, and ground truth, so misclassifications
are easy to spot visually.

The biggest practical challenge was balancing training speed against
accuracy: training the full MobileNetV2 from scratch would have needed far
more data and compute, so transfer learning was the right call — it reaches
strong accuracy (typically 97%+ test accuracy with this setup) in a fraction
of the time. The main engineering decision was the freeze-then-fine-tune
strategy, which avoids catastrophic forgetting of ImageNet features while
still letting the model specialize.

The trained model is saved in both native Keras and H5 formats, with a
separate `inference.py` script so it can be reused on arbitrary new images
without retraining.
