"""
============================================================
  Handwritten Character Recognition — CodeAlpha Internship
  Task 3: CNN on MNIST Dataset
  Author: [Your Name]
============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
#  1. CONFIGURATION
# ─────────────────────────────────────────────
IMG_ROWS, IMG_COLS = 28, 28
NUM_CLASSES        = 10
BATCH_SIZE         = 128
EPOCHS             = 20
MODEL_PATH         = "saved_model/mnist_cnn_model.h5"
CLASS_NAMES        = [str(i) for i in range(10)]   # digits 0–9

os.makedirs("saved_model", exist_ok=True)
os.makedirs("plots",       exist_ok=True)


# ─────────────────────────────────────────────
#  2. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
def load_and_preprocess():
    print("\n[1/5] Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape → (samples, height, width, channels)
    X_train = X_train.reshape(-1, IMG_ROWS, IMG_COLS, 1).astype("float32") / 255.0
    X_test  = X_test .reshape(-1, IMG_ROWS, IMG_COLS, 1).astype("float32") / 255.0

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

    print(f"    Train samples : {X_train.shape[0]}")
    print(f"    Test  samples : {X_test.shape[0]}")
    print(f"    Image shape   : {X_train.shape[1:]}")

    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat


# ─────────────────────────────────────────────
#  3. VISUALISE SAMPLE IMAGES
# ─────────────────────────────────────────────
def visualise_samples(X_train, y_train):
    print("\n[2/5] Saving sample images plot...")
    fig, axes = plt.subplots(4, 8, figsize=(14, 7))
    fig.suptitle("Sample MNIST Images", fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[i].reshape(IMG_ROWS, IMG_COLS), cmap="gray")
        ax.set_title(f"Label: {y_train[i]}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("plots/sample_images.png", dpi=150)
    plt.close()
    print("    Saved → plots/sample_images.png")


# ─────────────────────────────────────────────
#  4. BUILD CNN MODEL
# ─────────────────────────────────────────────
def build_model():
    print("\n[3/5] Building CNN model...")
    model = Sequential([
        # ── Block 1 ──────────────────────────
        Conv2D(32, (3, 3), activation="relu", padding="same",
               input_shape=(IMG_ROWS, IMG_COLS, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # ── Block 2 ──────────────────────────
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # ── Block 3 ──────────────────────────
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # ── Classifier ───────────────────────
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ─────────────────────────────────────────────
#  5. TRAIN MODEL
# ─────────────────────────────────────────────
def train_model(model, X_train, y_train_cat, X_test, y_test_cat):
    print("\n[4/5] Training model...")

    # Data augmentation (light — MNIST digits)
    datagen = ImageDataGenerator(
        rotation_range=8,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    datagen.fit(X_train)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test_cat),
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────
#  6. EVALUATE & VISUALISE RESULTS
# ─────────────────────────────────────────────
def evaluate_and_visualise(model, history, X_test, y_test, y_test_cat):
    print("\n[5/5] Evaluating model...")

    # ── Test accuracy ────────────────────────
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n    Test Accuracy : {acc * 100:.2f}%")
    print(f"    Test Loss     : {loss:.4f}")

    # ── Predictions ──────────────────────────
    y_pred      = model.predict(X_test, verbose=0)
    y_pred_cls  = np.argmax(y_pred, axis=1)

    # ── Classification Report ─────────────────
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred_cls,
                                target_names=CLASS_NAMES))

    # ── Plot 1: Training curves ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=15, fontweight="bold")

    axes[0].plot(history.history["accuracy"],     label="Train Acc",  color="#2196F3")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    color="#4CAF50")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train Loss", color="#F44336")
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#FF9800")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/training_history.png", dpi=150)
    plt.close()
    print("    Saved → plots/training_history.png")

    # ── Plot 2: Confusion Matrix ──────────────
    cm = confusion_matrix(y_test, y_pred_cls)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix", fontsize=15, fontweight="bold")
    plt.xlabel("Predicted Label"); plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=150)
    plt.close()
    print("    Saved → plots/confusion_matrix.png")

    # ── Plot 3: Sample predictions ────────────
    fig, axes = plt.subplots(4, 8, figsize=(14, 7))
    fig.suptitle("Model Predictions (Green=Correct, Red=Wrong)", fontsize=13, fontweight="bold")

    indices = np.random.choice(len(X_test), 32, replace=False)
    for ax, idx in zip(axes.flat, indices):
        ax.imshow(X_test[idx].reshape(IMG_ROWS, IMG_COLS), cmap="gray")
        color = "green" if y_pred_cls[idx] == y_test[idx] else "red"
        ax.set_title(f"P:{y_pred_cls[idx]} T:{y_test[idx]}", fontsize=7, color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("plots/predictions.png", dpi=150)
    plt.close()
    print("    Saved → plots/predictions.png")

    return acc, loss


# ─────────────────────────────────────────────
#  7. PREDICT A SINGLE IMAGE (DEMO)
# ─────────────────────────────────────────────
def predict_single(model, X_test, y_test, index=0):
    img   = X_test[index].reshape(1, IMG_ROWS, IMG_COLS, 1)
    probs = model.predict(img, verbose=0)[0]
    pred  = np.argmax(probs)
    conf  = probs[pred] * 100

    plt.figure(figsize=(6, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[index].reshape(IMG_ROWS, IMG_COLS), cmap="gray")
    plt.title(f"True: {y_test[index]}  |  Predicted: {pred}\nConfidence: {conf:.1f}%",
              color="green" if pred == y_test[index] else "red")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    bars = plt.bar(CLASS_NAMES, probs * 100, color="#2196F3", edgecolor="black")
    bars[pred].set_color("#4CAF50")
    plt.xlabel("Digit Class"); plt.ylabel("Confidence (%)")
    plt.title("Class Probabilities"); plt.ylim(0, 110)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/single_prediction.png", dpi=150)
    plt.close()
    print(f"    Single prediction → Predicted: {pred}  True: {y_test[index]}  "
          f"Confidence: {conf:.1f}%")
    print("    Saved → plots/single_prediction.png")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Handwritten Character Recognition — CodeAlpha Task 3")
    print("=" * 60)

    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = load_and_preprocess()
    visualise_samples(X_train, y_train)
    model   = build_model()
    history = train_model(model, X_train, y_train_cat, X_test, y_test_cat)
    acc, _  = evaluate_and_visualise(model, history, X_test, y_test, y_test_cat)
    predict_single(model, X_test, y_test, index=7)

    print("\n" + "=" * 60)
    print(f"  ✅  Final Test Accuracy : {acc * 100:.2f}%")
    print(f"  💾  Model saved at     : {MODEL_PATH}")
    print(f"  📊  Plots saved in     : plots/")
    print("=" * 60)
