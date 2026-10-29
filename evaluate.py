"""Confusion matrix + precision/recall per class (use these in your report/PPT)."""
import numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset", validation_split=0.2, subset="validation", seed=42,
    image_size=(224, 224), batch_size=32, label_mode="categorical", shuffle=False)
model = tf.keras.models.load_model("mask_model.keras")

y_true = np.concatenate([y.numpy().argmax(1) for _, y in val_ds])
y_pred = model.predict(val_ds).argmax(1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=val_ds.class_names))
