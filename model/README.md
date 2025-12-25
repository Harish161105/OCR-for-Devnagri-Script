

Devanagari OCR – Model Module

## Overview

This folder contains the complete machine learning pipeline for the **Devanagari Handwritten Optical Character Recognition (OCR)** system. The model is implemented using a **Convolutional Neural Network (CNN)** and is trained to recognize handwritten Devanagari characters and digits.



## Dataset

* **Name**: Devanagari Handwritten Character Dataset
* **Source**: UCI Machine Learning Repository / Kaggle
* **Classes**: 46

  * 36 Devanagari characters
  * 10 digits (0–9)
* **Image Size**: 32 × 32 pixels
* **Color Mode**: Grayscale

---

## Preprocessing Pipeline

The following preprocessing steps are applied before training and inference:

1. Conversion to grayscale
2. Background normalization (black text on white background)
3. Image resizing while preserving aspect ratio
4. Centering using padding
5. Pixel value normalization (0–1 range)

Preprocessing logic is implemented in:

```
preprocessing.py

```

---

## Model Architecture

The CNN model consists of:

* Input Layer: 32 × 32 grayscale image
* Convolutional Layers with ReLU activation
* Max Pooling layers
* Dropout layers for regularization
* Fully connected Dense layers
* Output layer with linear activation (logits)

---

## Training Details

* **Optimizer**: Adam
* **Loss Function**: Sparse Categorical Crossentropy (from logits)
* **Batch Size**: 32
* **Epochs**: 10
* **Framework**: TensorFlow / Keras

Training code is implemented in:

```
cnn.py
```

---

## Model Files

| File               | Description                              |
| ------------------ | ---------------------------------------- |
| `cnn.py`           | CNN model definition and training script |
| `preprocessing.py` | Image preprocessing utilities            |
| `testing.py`       | Model testing and prediction script      |
| `ocr_model2.h5`    | Trained CNN model                        |

---

## Evaluation

* Model performance is evaluated using training and validation accuracy
* Loss and accuracy curves are plotted after training
* The model performs well on clean, centered handwritten characters

---

## Limitations

* Trained on isolated characters only
* Sensitive to heavy noise and uncentered input
* Not designed for full word or sentence recognition

---

## Usage

To load the trained model:

```python
import tensorflow as tf
model = tf.keras.models.load_model("ocr_model2.h5")
```



## Author

**Harish Phad**
Project-Based Learning (PBL), Second Year
Devanagari OCR using CNN

---




