import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define dataset path
DATASET_PATH = r"C:\Users\abdul\OneDrive\Desktop\Dataset"  # Main folder with subfolders like 'basmati', 'jasmine', etc.

varieties = sorted(os.listdir(DATASET_PATH))
print("🔹 Found varieties:", varieties)
import os

# def print_directory_structure(path, level=0):
#     if os.path.isdir(path):
#         print("  " * level + f"[DIR] {os.path.basename(path)}")
#         for item in os.listdir(path):
#             print_directory_structure(os.path.join(path, item), level + 1)

# DATASET_PATH = r"C:\Users\abdul\OneDrive\Desktop\Dataset"

# print("🔹 Displaying folder structure:")
# print_directory_structure(DATASET_PATH)



IMG_SIZE = (256, 256)  # Resized image size
BATCH_SIZE = 8  # Training batch size

# Load images and masks
def load_images_and_masks(dataset_path):
    images = []
    masks = []
    labels = []
    
    class_folders = os.listdir(dataset_path)  # Get all subfolders (rice varieties)
    
    for class_label, class_folder in enumerate(class_folders):
        class_path = os.path.join(dataset_path, class_folder)
        image_path = os.path.join(class_path, "images")  # Folder containing images
        mask_path = os.path.join(class_path, "masks")  # Folder containing masks

        # Check if the image and mask directories exist
        if not os.path.exists(image_path):
            print(f"⚠️ Image directory not found: {image_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"⚠️ Mask directory not found: {mask_path}")
            continue

        for filename in os.listdir(image_path):
            image_file = os.path.join(image_path, filename)
            mask_filename = filename.replace(".jpeg", "_mask.jpeg")  # Construct mask filename
            mask_file = os.path.join(mask_path, mask_filename)  # Path to mask file
            
            if not os.path.exists(mask_file):  # Ensure mask exists for the image
                print(f"⚠️ Mask not found for image: {image_file} (expected: {mask_file})")
                continue

            try:
                img = load_img(image_file, target_size=IMG_SIZE)  # Load and resize image
                img = img_to_array(img) / 255.0  # Normalize image
                
                mask = load_img(mask_file, target_size=IMG_SIZE, color_mode="grayscale")  # Load mask
                mask = img_to_array(mask) / 255.0  # Normalize mask
                
                images.append(img)
                masks.append(mask)
                labels.append(class_label)  # Store class label (optional)
            except Exception as e:
                print(f"⚠️ Error loading {image_file} or {mask_file}: {e}")

    return np.array(images), np.array(masks), np.array(labels)

# Load data
X, Y, labels = load_images_and_masks(DATASET_PATH)
print(f"Loaded {len(X)} images and {len(Y)} masks.")

# Split into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# U-Net Model
def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)  # Single-channel output (binary mask)

    model = Model(inputs, outputs)
    return model

# Compile Model
model = build_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define Callbacks
checkpoint = ModelCheckpoint('best_unet_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
csv_logger = CSVLogger('training_log.csv', append=True)

# Train Model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=BATCH_SIZE,
    epochs=50,
    callbacks=[checkpoint, early_stopping, reduce_lr, csv_logger]
)

# Save Final Model
model.save('final_unet_model.h5')

# Plot Training Results
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("U-Net Training Loss Curve")
plt.show()
