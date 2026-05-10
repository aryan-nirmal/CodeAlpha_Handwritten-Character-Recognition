"""
predict_custom.py
─────────────────
Use your trained model to predict a digit from ANY image file.

Usage:
    python predict_custom.py --image path/to/your/image.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

MODEL_PATH  = "saved_model/mnist_cnn_model.h5"
CLASS_NAMES = [str(i) for i in range(10)]


def preprocess_image(path: str) -> np.ndarray:
    """Load any image, convert to 28×28 grayscale, invert if needed."""
    img = Image.open(path).convert("L")          # grayscale
    img = ImageOps.invert(img)                   # make digit white on black
    img = img.resize((28, 28), Image.LANCZOS)    # resize to MNIST size
    arr = np.array(img, dtype="float32") / 255.0 # normalise
    return arr.reshape(1, 28, 28, 1)


def predict(image_path: str):
    print(f"Loading model from '{MODEL_PATH}' ...")
    model = load_model(MODEL_PATH)

    print(f"Processing image '{image_path}' ...")
    img_array = preprocess_image(image_path)

    probs = model.predict(img_array, verbose=0)[0]
    pred  = np.argmax(probs)
    conf  = probs[pred] * 100

    print(f"\n  Predicted Digit : {pred}")
    print(f"  Confidence      : {conf:.1f}%")

    # ── Visualise ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Custom Image Prediction", fontsize=14, fontweight="bold")

    axes[0].imshow(img_array.reshape(28, 28), cmap="gray")
    axes[0].set_title(f"Predicted: {pred}  ({conf:.1f}%)", fontsize=12)
    axes[0].axis("off")

    bars = axes[1].bar(CLASS_NAMES, probs * 100, color="#2196F3", edgecolor="black")
    bars[pred].set_color("#4CAF50")
    axes[1].set_xlabel("Digit Class")
    axes[1].set_ylabel("Confidence (%)")
    axes[1].set_title("Class Probabilities")
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/custom_prediction.png", dpi=150)
    plt.show()
    print("  Plot saved → plots/custom_prediction.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict digit from a custom image.")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image (jpg/png/bmp)")
    args = parser.parse_args()
    predict(args.image)
