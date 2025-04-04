import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import numpy as np

# Input and output folders
INPUT_DIR = r"C:\Users\abdul\OneDrive\Desktop\Major-Project\Rice-unet\SegmentedDataset"
OUTPUT_DIR = r"C:\Users\abdul\OneDrive\Desktop\Major-Project\Rice-unet\Augmented"

# Number of augmented images per original
AUG_PER_IMAGE = 20

# Define augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each class folder
for class_name in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, class_name)
    save_path = os.path.join(OUTPUT_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    os.makedirs(save_path, exist_ok=True)

    print(f"Processing class: {class_name}")
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in image_files:
        img_path = os.path.join(class_path, img_name)
        img = load_img(img_path)  # Load image
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Shape: (1, h, w, 3)

        # Generate augmented images
        prefix = os.path.splitext(img_name)[0]
        aug_iter = datagen.flow(x, batch_size=1, save_to_dir=save_path,
                                save_prefix=prefix, save_format='jpg')

        # Save a fixed number of augmented images
        for i in range(AUG_PER_IMAGE):
            next(aug_iter)

    print(f"✅ Done: {class_name}")

print("\n🎉 All classes processed and augmented!")
