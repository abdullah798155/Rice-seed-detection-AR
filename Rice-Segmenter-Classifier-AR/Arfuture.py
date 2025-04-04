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

health_benefits = {
    'Basmati': [
        "✦ Low Glycemic Index",
        "✦ Heart friendly",
        "✦ Rich aroma & fiber"
    ],
    'Gobindobhog': [
        "✦ Boosts energy",
        "✦ Used in Ayurveda",
        "✦ Aromatic & digestible"
    ],
    'Kolam': [
        "✦ Light on stomach",
        "✦ Perfect daily rice",
        "✦ Quick to cook"
    ]
}

# Webcam setup
cap = cv2.VideoCapture(0)
cv2.namedWindow("Rice AR", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Rice AR", 900, 600)

frame_count = 0
variety_label = "No seed detected"
last_variety = None

def draw_transparent_box(img, top_left, bottom_right, alpha=0.3, color=(0, 0, 0)):
    """Draw semi-transparent rectangle on image"""
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, UNET_IMG_SIZE)
    input_img = resized / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    # U-Net prediction
    pred_mask = unet_model.predict(input_img, verbose=0)[0, :, :, 0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    mask_area = np.sum(binary_mask)

    # Overlay
    green = np.zeros_like(resized)
    green[:, :, 1] = 255
    mask_3ch = np.stack([binary_mask]*3, axis=-1)
    overlay = np.where(mask_3ch == 1, cv2.addWeighted(resized, 0.6, green, 0.4, 0), resized)

    # Upscale and contour
    overlay_full = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))
    binary_mask_large = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(binary_mask_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_full, contours, -1, (0, 255, 255), 2)
    print("mask area:", mask_area)
    if frame_count % 10 == 0:
        if mask_area:
            masked_img = resized.copy()
            masked_img[binary_mask == 0] = 0

            classified_img = cv2.resize(masked_img, MOBILENET_IMG_SIZE)
            classified_img = classified_img / 255.0
            classified_img = np.expand_dims(classified_img, axis=0)

            prediction = mobilenet_model.predict(classified_img, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            variety_label = f"{class_labels[predicted_class]} ({confidence*100:.1f}%)"
            print("confidence: ", confidence)
            last_variety = class_labels[predicted_class]
        else:
            variety_label = "No seed detected"

    # --- Top-left label with transparent background ---
    overlay_full = draw_transparent_box(overlay_full, (20, 20), (580, 70), alpha=0.4)
    cv2.putText(
        overlay_full,
        f"Rice Variety: {variety_label}",
        (30, 60),
        cv2.FONT_HERSHEY_PLAIN,
        2.2,
        (0, 255, 0),
        2
    )

    # --- Bottom-right health benefits box ---
    if last_variety and last_variety in health_benefits:
        start_x = overlay_full.shape[1] - 310
        start_y = overlay_full.shape[0] - 100
        overlay_full = draw_transparent_box(overlay_full, (start_x - 20, start_y - 30), (start_x + 280, start_y + 80), alpha=0.35)

        cv2.putText(
            overlay_full,
            "Health Benefits:",
            (start_x, start_y),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 255, 0),
            2
        )

        for i, point in enumerate(health_benefits[last_variety]):
            cv2.putText(
                overlay_full,
                point,
                (start_x, start_y + 25 + i * 18),
                cv2.FONT_HERSHEY_PLAIN,
                1.2,
                (0, 220, 0),
                1
            )

    cv2.imshow("Rice AR", overlay_full)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
