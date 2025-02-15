import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tkinter as tk
from tkinter import filedialog

# Paths to models
unet_model_path = r"C:\Users\abdul\OneDrive\Desktop\Rice-unet\final_unet_model.h5"
resnet_model_path = r"C:\Users\abdul\OneDrive\Desktop\Rice-unet\resnet50_rice_classifier.h5"

# Load U-Net and ResNet50 models
unet_model = load_model(unet_model_path)
resnet_model = load_model(resnet_model_path)

# Class labels for rice variety classification
class_labels = ['Basmati', 'Gobindobhog','Kolam']  # Update if needed

# Image size (should match U-Net and ResNet input size)
IMG_SIZE = (256, 256)

# Function to upload image
def upload_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    return file_path

# Function to segment the rice seeds using U-Net
def segment_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict mask
    predicted_mask = unet_model.predict(img_array)[0, :, :, 0]
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255  # Binary mask

    # Load original image
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, IMG_SIZE)

    # Apply mask
    segmented_img = cv2.bitwise_and(original_img, original_img, mask=predicted_mask)

    # Save segmented image temporarily
    segmented_image_path = "segmented_image.jpg"
    cv2.imwrite(segmented_image_path, segmented_img)

    return segmented_image_path

# Function to classify rice variety using ResNet50
def predict_variety(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = resnet_model.predict(img_array)
    predicted_class = np.argmax(predictions)
    variety = class_labels[predicted_class]

    return variety

# Main function
def main():
    print("Select an image to predict:")
    image_path = upload_image()

    if not image_path:
        print("No image selected.")
        return

    print(f"Segmenting rice seeds from: {image_path}")
    segmented_image_path = segment_image(image_path)

    print(f"Segmented image saved at: {segmented_image_path}")
    
    print("Predicting rice variety...")
    variety = predict_variety(segmented_image_path)
    
    print(f"Predicted Rice Variety: {variety}")

# Run the script
if __name__ == "__main__":
    main()
