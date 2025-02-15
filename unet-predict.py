import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the trained model
MODEL_PATH = r'C:\Users\abdul\OneDrive\Desktop\Rice-unet\final_unet_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Path to the image you want to predict
IMAGE_PATH = r"C:\Users\abdul\OneDrive\Desktop\Dataset\Gobindobhog\images\gobindobhog2.jpeg"
image = load_and_preprocess_image(IMAGE_PATH)

# Predict the mask
predicted_mask = model.predict(image)

# Debug: Inspect raw predictions
print("Predicted mask shape:", predicted_mask.shape)
print("Raw predictions min:", predicted_mask.min())
print("Raw predictions max:", predicted_mask.max())

# Visualize raw prediction map (using hot colormap for better clarity)
plt.imshow(predicted_mask[0, :, :, 0], cmap='hot')
plt.colorbar()
plt.title("Raw Prediction Map")
plt.show()

# Post-process the mask (adjust threshold if needed)
predicted_mask = (predicted_mask > 0.3).astype(np.uint8)  # Try a lower threshold

# Scale the mask back for visualization
predicted_mask_scaled = (predicted_mask[0, :, :, 0] * 255).astype(np.uint8)

# Display the original image and the predicted mask
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image[0])
plt.axis('off')

# Predicted Mask
plt.subplot(1, 2, 2)
plt.title('Predicted Mask')
plt.imshow(predicted_mask_scaled, cmap='gray')
plt.axis('off')

plt.show()
