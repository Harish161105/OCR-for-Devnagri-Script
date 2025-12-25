import tensorflow as tf
import numpy as np
import cv2
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model
model = tf.keras.models.load_model("devanagari_ocr_model.h5")  # Update with your model filename

# Define class labels
class_labels = {
    0: 'अ', 1: 'आ', 2: 'इ', 3: 'ई', 4: 'उ', 5: 'ऊ', 6: 'ऋ', 7: 'ए', 
    8: 'ऐ', 9: 'ओ', 10: 'औ', 11: 'क', 12: 'ख', 13: 'ग', 14: 'घ', 15: 'ङ',
    16: 'च', 17: 'छ', 18: 'ज', 19: 'झ', 20: 'ञ', 21: 'ट', 22: 'ठ', 23: 'ड',
    24: 'ढ', 25: 'ण', 26: 'त', 27: 'थ', 28: 'द', 29: 'ध', 30: 'न', 31: 'प',
    32: 'फ', 33: 'ब', 34: 'भ', 35: 'म', 36: 'य', 37: 'र', 38: 'ल', 39: 'व',
    40: 'श', 41: 'ष', 42: 'स', 43: 'ह', 44: 'क्ष', 45: 'त्र'
}

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    img = cv2.resize(img, (32, 32))  # Resize to match model input size
    
    # Convert grayscale (1 channel) to RGB (3 channels)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, 32, 32, 3)  # Reshape for model input (3 channels)
    
    return img
# Load and preprocess the image
image_path = "D:\\OCR DEVNAGRI\\testfolder\\190.png"  # Update with your image path
input_image = preprocess_image(image_path)

# Get model prediction
predictions = model.predict(input_image)
predicted_class = np.argmax(predictions)

# Convert to Devanagari character
predicted_char = class_labels.get(predicted_class, "Unknown")

print(f"Predicted Character: {predicted_char}")
