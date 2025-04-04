import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ✅ Set dataset path (Use full path)
DATASET_PATH = r'C:\Users\abdul\OneDrive\Desktop\Major-Project\Rice-unet\Augmented'

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
NUM_CLASSES = 3  # Since you have basmati, gobindobhog, kolam

# Data augmentation & loading
train_datagen = ImageDataGenerator(rescale=1.0/255)  # No validation split

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,  # Root dataset directory
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load MobileNetV2 (pretrained)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(NUM_CLASSES, activation='softmax')(x)

# Create final model
model = keras.Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model (No validation)
model.fit(train_generator, epochs=10)

# Save as .h5
model.save('mobilenetv2_augmented.h5')
print("Model saved as mobilenetv2_rice.h5! 🎉")
