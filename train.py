import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

print("TRAINING SCRIPT STARTED", flush=True)

DATA_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42

print("Checking dataset...", flush=True)

if not os.path.exists(DATA_DIR):
    print("ERROR: dataset folder not found!", flush=True)
    exit()

class_folders = [
    f for f in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, f))
]

print("Found classes:", class_folders, flush=True)

if len(class_folders) != 3:
    print("WARNING: Expected 3 classes.", flush=True)

print("Loading training dataset...", flush=True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical"
)

print("Loading validation dataset...", flush=True)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical"
)

class_names = train_ds.class_names

print("Classes:", class_names, flush=True)

with open("labels.txt", "w") as f:
    f.write("\n".join(class_names))

counts = []

for c in class_names:
    folder = os.path.join(DATA_DIR, c)
    count = len([
        file for file in os.listdir(folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    counts.append(count)

total = sum(counts)

class_weight = {
    i: (total / (len(counts) * n)) ** 0.5
    for i, n in enumerate(counts)
    if n > 0
}

print("Counts:", dict(zip(class_names, counts)), flush=True)
print("Weights:", class_weight, flush=True)

print("Preparing data augmentation...", flush=True)

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2, value_range=(0, 255))
])

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(
    lambda x, y: (augment(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)

print("Loading MobileNetV2...", flush=True)

base = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base.trainable = False

print("Building model...", flush=True)

inputs = layers.Input(shape=IMG_SIZE + (3,))

x = layers.Rescaling(
    1 / 127.5,
    offset=-1
)(inputs)

x = base(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting initial training...", flush=True)

h1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    class_weight=class_weight
)

print("Initial training completed.", flush=True)

print("Starting fine-tuning...", flush=True)

base.trainable = True

for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

h2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weight
)

print("Fine-tuning completed.", flush=True)

print("Saving model...", flush=True)

model.save("mask_model.keras")

acc = h1.history["accuracy"] + h2.history["accuracy"]
val = h1.history["val_accuracy"] + h2.history["val_accuracy"]

plt.plot(acc, label="train acc")
plt.plot(val, label="val acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("training_curve.png", dpi=150)
plt.close()

print("Saved mask_model.keras", flush=True)
print("Saved labels.txt", flush=True)
print("Saved training_curve.png", flush=True)
print("TRAINING COMPLETED SUCCESSFULLY", flush=True)

