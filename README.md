# 📧 SMS Spam Classifier using Deep Learning

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![NLP](https://img.shields.io/badge/NLP-Spam%20Detection-green)

This repository contains an end-to-end Natural Language Processing (NLP) project that classifies SMS messages as **Ham** (legitimate) or **Spam**. The project compares traditional Dense networks, Recurrent Neural Networks (Bi-LSTM), and Transfer Learning approaches.

## 📊 Dataset
The project uses a spam dataset (CSV format) containing raw text messages labeled as `ham` or `spam`.
- **Training Samples:** 4,457
- **Average Words per Message:** 16
- **Vocabulary Size:** ~18,392

## 🏗️ Model Architectures
The project implements and compares three distinct neural network architectures:

1. **Dense Model (Custom Embedding):**
   - Text Vectorization layer.
   - Custom Embedding layer (128 dimensions).
   - Global Average Pooling with a Dense ReLU layer.

2. **Bi-LSTM Model:**
   - Designed to capture sequential dependencies in text.
   - Two layers of Bidirectional LSTMs (64 units each).
   - Dropout (0.1) for regularization.

3. **Transfer Learning (Universal Sentence Encoder):**
   - Utilizes Google's pre-trained **Universal Sentence Encoder (USE)** from TensorFlow Hub.
   - Provides fixed-length 512-dimensional embeddings regardless of message length.

## 📈 Performance Results
All models were trained for 5 epochs using the Adam optimizer and Binary Crossentropy loss.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Embedding** | 97.85% | 0.956 | 0.879 | 0.916 |
| **Bi-LSTM** | 98.12% | 0.932 | 0.926 | 0.929 |
| **Transfer Learning (USE)** | **98.39%** | **0.952** | **0.926** | **0.939** |

*Note: Transfer Learning via USE provided the most balanced performance across all metrics.*

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow, Keras
- **NLP:** TensorFlow Hub (Universal Sentence Encoder), TextVectorization
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## 🚀 How to Run
1. **Prepare the Data:** Ensure your `spam dataset.csv` is in the working directory.
2. **Install Requirements:**
3. Execute: Run the Spam_Classifier (1).ipynb notebook in Google Colab or a local Jupyter environment.

📂 Project Structure
Spam_Classifier (1).ipynb: Full pipeline including EDA, preprocessing, model training, and evaluation.

spam dataset.csv: The raw message dataset.

Developed as a comparison of NLP methodologies for text classification.
   ```bash
   pip install tensorflow tensorflow-hub pandas numpy matplotlib seaborn scikit-learn
