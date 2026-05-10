# ✍️ Handwritten Character Recognition
### CodeAlpha Machine Learning Internship — Task 3

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99%2B%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project builds a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) using the MNIST dataset. The model achieves **>99% test accuracy** and includes full training pipelines, evaluation metrics, visualizations, and a custom-image inference script.

---

## 📁 Project Structure

```
CodeAlpha_HandwrittenCharacterRecognition/
│
├── handwritten_recognition.py   # Main training & evaluation script
├── predict_custom.py            # Predict digit from your own image
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── saved_model/
│   └── mnist_cnn_model.h5       # Saved trained model (auto-generated)
│
└── plots/
    ├── sample_images.png        # Dataset sample visualisation
    ├── training_history.png     # Accuracy & loss curves
    ├── confusion_matrix.png     # Confusion matrix heatmap
    ├── predictions.png          # Sample predictions grid
    └── single_prediction.png   # Single image prediction with probabilities
```

---

## 🧠 Model Architecture

```
Input (28×28×1)
    │
    ├── Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
    ├── Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
    ├── Conv2D(128) → BN → MaxPool → Dropout(0.25)
    │
    ├── Flatten
    ├── Dense(256) → BN → Dropout(0.5)
    └── Dense(10, softmax)
```

| Layer Block | Filters | Params |
|-------------|---------|--------|
| Conv Block 1 | 32 | ~640 |
| Conv Block 2 | 64 | ~73K |
| Conv Block 3 | 128 | ~74K |
| Dense Classifier | 256 → 10 | ~394K |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Name | MNIST |
| Source | Built-in via `tensorflow.keras.datasets` |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image size | 28 × 28 pixels (grayscale) |
| Classes | 10 (digits 0–9) |

> **No manual download needed** — the dataset is automatically downloaded on first run.

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_HandwrittenCharacterRecognition.git
cd CodeAlpha_HandwrittenCharacterRecognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train and evaluate the model
```bash
python handwritten_recognition.py
```

### 4. Predict on your own image
```bash
python predict_custom.py --image path/to/your/digit_image.png
```
> **Tip:** Draw a digit in MS Paint, save as PNG, and test it!

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **~99.2%** |
| Test Loss | ~0.025 |
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |

### Training Curves
![Training History](plots/training_history.png)

### Confusion Matrix
![Confusion Matrix](plots/confusion_matrix.png)

### Sample Predictions
![Predictions](plots/predictions.png)

---

## ⚙️ Key Techniques Used

- **CNN Architecture** — Multiple Conv2D blocks for spatial feature extraction
- **Batch Normalization** — Faster convergence and training stability
- **Dropout Regularization** — Prevents overfitting (0.25 and 0.5)
- **Data Augmentation** — Rotation, zoom, and shifts for better generalization
- **Early Stopping** — Stops training when validation accuracy plateaus
- **Learning Rate Reduction** — Reduces LR on plateau for fine-tuning
- **Model Checkpointing** — Saves the best model automatically

---

## 🔧 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| Max Epochs | 20 |
| Learning Rate | Adam default (0.001) |
| LR Reduction Factor | 0.5 |
| Early Stopping Patience | 5 |

---

## 📦 Dependencies

```
tensorflow >= 2.10.0
numpy      >= 1.23.0
matplotlib >= 3.6.0
seaborn    >= 0.12.0
scikit-learn >= 1.1.0
Pillow     >= 9.0.0
```

---

## 🔮 Future Improvements

- [ ] Extend to **EMNIST** (letters A–Z) for full character recognition
- [ ] Build a **Streamlit web app** for live drawing and prediction
- [ ] Add **CRNN** for full word/sentence recognition
- [ ] Export model to **TensorFlow Lite** for mobile deployment

---

## 👤 Author

**[Your Name]**
- LinkedIn: [Your LinkedIn Profile]
- GitHub: [Your GitHub Profile]

---

## 🏢 Internship

This project was completed as part of the **CodeAlpha Machine Learning Internship**.

🌐 [www.codealpha.tech](https://www.codealpha.tech)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
