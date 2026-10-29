import os
import tensorflow as tf

print("STEP 1: Script started", flush=True)

print("STEP 2: Loading Keras model...", flush=True)

model = tf.keras.models.load_model("mask_model.keras")

print("STEP 3: Keras model loaded", flush=True)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

print("STEP 4: Converter created", flush=True)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("STEP 5: Starting TFLite conversion...", flush=True)

tflite = converter.convert()

print("STEP 6: Conversion completed", flush=True)

with open("mask_model.tflite", "wb") as f:
    f.write(tflite)

print("STEP 7: TFLite file saved", flush=True)

keras_size = os.path.getsize("mask_model.keras") / 1e6
tflite_size = len(tflite) / 1e6

print(f"Keras : {keras_size:.1f} MB", flush=True)
print(f"TFLite: {tflite_size:.1f} MB", flush=True)

print("CONVERSION SUCCESSFUL", flush=True)