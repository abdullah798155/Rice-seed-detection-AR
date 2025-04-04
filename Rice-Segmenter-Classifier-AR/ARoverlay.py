import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load models
unet_model_path = r"C:\Users\abdul\OneDrive\Desktop\Major-Project\Rice-unet\final_unet_model.h5"
mobilenet_model_path = r"C:\Users\abdul\OneDrive\Desktop\Major-Project\Rice-unet\NewModel\mobilenetv2_augmented.h5"

unet_model = load_model(unet_model_path)
mobilenet_model = load_model(mobilenet_model_path)

# Constants
UNET_IMG_SIZE = (256, 256)
MOBILENET_IMG_SIZE = (224, 224)
class_labels = ['Basmati', 'Gobindobhog', 'Kolam']

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Rice AR", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Rice AR", 900, 600)

frame_count = 0
variety_label = "No seed detected"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for U-Net
    resized = cv2.resize(frame, UNET_IMG_SIZE)
    input_img = resized / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    # Predict mask using U-Net
    pred_mask = unet_model.predict(input_img, verbose=0)[0, :, :, 0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    mask_area = np.sum(binary_mask)

    # Green overlay for visualization
    green = np.zeros_like(resized)
    green[:, :, 1] = 255
    mask_3ch = np.stack([binary_mask]*3, axis=-1)
    overlay = np.where(mask_3ch == 1, cv2.addWeighted(resized, 0.6, green, 0.4, 0), resized)

    # Upscale overlay to original frame size
    overlay_full = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))

    # Show mask area (debugging)
    # print("Mask area:", mask_area)
    print("mask area: ",mask_area)
    # Classification if enough seed is detected
    if frame_count % 10 == 0:
        if mask_area > 1500:
            # Mask out background before classification
            masked_img = resized.copy()
            masked_img[binary_mask == 0] = 0

            # Resize and normalize
            classified_img = cv2.resize(masked_img, MOBILENET_IMG_SIZE)
            classified_img = classified_img / 255.0
            classified_img = np.expand_dims(classified_img, axis=0)

            # Predict variety
            prediction = mobilenet_model.predict(classified_img, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            print("confiddence: ",confidence)
            variety_label = f"{class_labels[predicted_class]} ({confidence*100:.1f}%)"
           
        else:
            variety_label = "No seed detected"

    # Draw label
    cv2.putText(
        overlay_full,
        f"Rice Variety: {variety_label}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3
    )

    # Show frame
    cv2.imshow("Rice AR", overlay_full)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
