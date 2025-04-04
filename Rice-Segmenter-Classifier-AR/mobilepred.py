import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
MODEL_PATH = "mobilenetv2_rice.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (must match training folder names)
class_names = ['basmati', 'gobindobhog', 'kolam']  # Change if needed

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for model input
    return img_array

# Predict function
def predict_rice(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get index of max confidence
    confidence = np.max(predictions)  # Get confidence score

    print(f"Predicted: {class_names[predicted_class]} (Confidence: {confidence:.2f})")

# Test the prediction on an image
IMAGE_PATH = r"C:\Users\abdul\OneDrive\Desktop\Major-Project\Rice-unet\SegmentedDataset\Kolam\kolam3.jpeg"  # Change this to an actual test image
predict_rice(IMAGE_PATH)
