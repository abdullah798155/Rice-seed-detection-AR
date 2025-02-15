import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Load trained model
model = load_model(r"C:\Users\abdul\OneDrive\Desktop\Rice-unet\resnet50_rice_classifier.h5")

# Class labels (should match folder names in SegmentedDataset)
class_labels = ['Basmati', 'Kolam', 'Gobindobhog']  # Update if needed

# Function to classify a new segmented image
def predict_variety(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    variety = class_labels[predicted_class]

    print(f"Predicted Variety: {variety}")

# Example usage
test_image = r"C:\Users\abdul\OneDrive\Desktop\SegmentedDataset\Kolam\kolam1.jpeg"
predict_variety(test_image)
