import cv2
import numpy as np
import os

def preprocess_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    if np.mean(img) < 127:
        img = cv2.bitwise_not(img) 
     
    img = cv2.resize(img, (32, 32))
    
    img = img / 255.0  
    
    return (img * 255).astype(np.uint8) 

def process_dataset(input_folder, output_folder):
    
    if not os.path.exists(output_folder):  
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                
                
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                
                processed_img = preprocess_image(input_path)
                cv2.imwrite(output_path, processed_img)


train_folder = "D:\OCR DEVNAGRI\datasets\devanagari+handwritten+character+dataset\DevanagariHandwrittenCharacterDataset\Train"  # Original training data
test_folder = "D:\OCR DEVNAGRI\datasets\devanagari+handwritten+character+dataset\DevanagariHandwrittenCharacterDataset\Test"  # Original testing data
new_train_folder = "D:\\OCR DEVNAGRI\\datasets\\processed for model 2\\train" # New processed training data
new_test_folder = "D:\\OCR DEVNAGRI\\datasets\\processed for model 2\\test" # New processed testing data


process_dataset(train_folder, new_train_folder)


print(" Processed images saved to new folders!")
