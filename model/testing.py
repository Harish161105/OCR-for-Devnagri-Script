import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import cv2
import os
sys.stdout.reconfigure(encoding='utf-8')

# Load trained model
model = tf.keras.models.load_model("D:/OCR DEVNAGRI/ocr_model2.h5")

# Define class names
class_names = [
    
    'क', 'ख', 'ग', 'घ', 'ङ',  
    'च', 'छ', 'ज', 'झ', 'ञ',  
    'ट', 'ठ', 'ड', 'ढ', 'ण',  
    'त', 'थ', 'द', 'ध', 'न',  
    'प', 'फ', 'ब', 'भ', 'म',  
    'य', 'र', 'ल', 'व',  
    'श', 'ष', 'स', 'ह',  
    'क्ष', 'त्र', 'ज्ञ',  
    '०', '१', '२', '३', '४', '५', '६', '७', '८', '९'  # Digits 0-9
]

def preprocess_image(image_path, output_size=(32, 32)):
    
    img = tf.keras.utils.load_img(image_path, target_size=(32, 32), color_mode='grayscale')
    
    

    return img

def predict_images(image_paths):
    """Processes and predicts classes for multiple images."""
    images = []

    for img_path in image_paths:
        processed_img = preprocess_image(img_path)
        if processed_img is not None:
            images.append(processed_img)
    
    if not images:
        print("No valid images processed.")
        return

    # Convert to numpy array and make batch
    images_array = np.array(images)
    
    # Get predictions
    predictions = model.predict(images_array)
    scores = tf.nn.softmax(predictions, axis=1)

    for i, img_path in enumerate(image_paths):
        class_idx = np.argmax(scores[i])
        confidence = 100 * np.max(scores[i])
        print(f"Image '{os.path.basename(img_path)}' most likely belongs to '{class_names[class_idx]}' with {confidence:.2f}% confidence.")

# Example: Predicting 5 images
image_paths = [
    "D:\\OCR DEVNAGRI\\testfolder\\don.jpeg",
   
]

predict_images(image_paths)
