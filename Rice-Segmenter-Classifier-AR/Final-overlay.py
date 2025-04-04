import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Paths to models
unet_model_path = r"C:\Users\abdul\OneDrive\Desktop\Major-Project\Rice-unet\final_unet_model.h5"
mobilenet_model_path = r"C:\Users\abdul\OneDrive\Desktop\Major-Project\Rice-unet\NewModel\mobilenetv2_augmented.h5"

# Load models
unet_model = load_model(unet_model_path)
mobilenet_model = load_model(mobilenet_model_path)

# Class labels
class_labels = ['Basmati', 'Gobindobhog', 'Kolam']

# Image sizes
UNET_IMG_SIZE = (256, 256)
MOBILENET_IMG_SIZE = (224, 224)

# Upload image
def upload_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg;*.png;*.jpeg")]
    )
    return file_path

# Segment image using U-Net and prepare both outputs
def segment_image(image_path):
    # Load and resize input for model
    img = load_img(image_path, target_size=UNET_IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict mask
    predicted_mask = unet_model.predict(img_array)[0, :, :, 0]
    binary_mask = (predicted_mask > 0.5).astype(np.uint8)

    # Load original image (OpenCV BGR)
    original_bgr = cv2.imread(image_path)
    original_bgr = cv2.resize(original_bgr, UNET_IMG_SIZE)

    # Create green overlay where mask is 1
    overlay = original_bgr.copy()
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255  # Green channel

    mask_3ch = np.stack([binary_mask]*3, axis=-1)  # Make mask 3-channel
    overlay = np.where(mask_3ch == 1, cv2.addWeighted(original_bgr, 0.6, green, 0.4, 0), original_bgr)

    # Create pure segmented output
    segmented_output = cv2.bitwise_and(original_bgr, original_bgr, mask=(binary_mask * 255).astype(np.uint8))

    # Save segmented version temporarily
    cv2.imwrite("segmented_image.jpg", segmented_output)

    return overlay, segmented_output, "segmented_image.jpg"

# Predict rice variety
def predict_variety(image_path):
    img = load_img(image_path, target_size=MOBILENET_IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = mobilenet_model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    variety = class_labels[predicted_class]

    print(f"Predicted Rice Variety: {variety} (Confidence: {confidence:.2f})")
    return variety, confidence

# Main function
def main():
    print("Select an image to predict:")
    image_path = upload_image()

    if not image_path:
        print("No image selected.")
        return

    print(f"Segmenting rice seeds from: {image_path}")
    overlay_img, segmented_img, segmented_image_path = segment_image(image_path)

    print("Predicting rice variety...")
    variety, confidence = predict_variety(segmented_image_path)

    # Convert images to RGB for matplotlib
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    segmented_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)

    # Display side-by-side
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(overlay_rgb)
    plt.title("Real time Overlay")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_rgb)
    plt.title("Segmented Output")
    plt.axis('off')

    plt.suptitle(f"Predicted Variety: {variety} | Confidence: {confidence:.2f}", fontsize=14, color='green')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Run
if __name__ == "__main__":
    main()
