from tensorflow import keras
import tensorflow as tf

# Load and convert U-Net
model_unet = keras.models.load_model(r'C:\Users\abdul\OneDrive\Desktop\Major-Project\Rice-unet\NewModel\final_unet_model.h5')
converter_unet = tf.lite.TFLiteConverter.from_keras_model(model_unet)
tflite_unet = converter_unet.convert()
open("unet.tflite", "wb").write(tflite_unet)

# Load and convert MobileNetV2
model_mnet = keras.models.load_model(r'C:\Users\abdul\OneDrive\Desktop\Major-Project\Rice-unet\NewModel\mobilenetv2_rice.h5')
converter_mnet = tf.lite.TFLiteConverter.from_keras_model(model_mnet)
tflite_mnet = converter_mnet.convert()
open("mobilenetv2_rice.tflite", "wb").write(tflite_mnet)
