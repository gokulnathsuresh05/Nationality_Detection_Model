import os
import csv
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==================================================
# PATHS
# ==================================================

IMAGES = r"F:\Nationality_Detection_Model\dataset\UTKFace\images"

TRAIN_CSV = r"F:\Nationality_Detection_Model\dataset\UTKFace\train.csv"
VAL_CSV = r"F:\Nationality_Detection_Model\dataset\UTKFace\val.csv"
TEST_CSV = r"F:\Nationality_Detection_Model\dataset\UTKFace\test.csv"

MODEL_FILE = r"F:\Nationality_Detection_Model\models\utkface_age_race.keras"
HISTORY_FILE = r"F:\Nationality_Detection_Model\results\utkface_training_history.json"

# ==================================================
# SETTINGS
# ==================================================

IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 30

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==================================================
# GPU CHECK
# ==================================================

print("")
print("============================================")
print(" HARDWARE CHECK")
print("============================================")
print("")

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("GPU detected:")
    
    for gpu in gpus:
        print(" ", gpu)

    try:

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

    except Exception as e:
        print("GPU memory-growth setup:", e)

else:

    print("No GPU detected.")
    print("Training will use CPU.")

print("")

# ==================================================
# LOAD CSV
# ==================================================

def load_csv(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return list(csv.DictReader(f))


train_rows = load_csv(TRAIN_CSV)
val_rows = load_csv(VAL_CSV)
test_rows = load_csv(TEST_CSV)

print("============================================")
print(" DATASET")
print("============================================")
print("")

print("Train:", len(train_rows))
print("Val  :", len(val_rows))
print("Test :", len(test_rows))
print("")

# ==================================================
# IMAGE LOADER
# ==================================================

def load_image(filename):

    path = os.path.join(
        IMAGES,
        filename
    )

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image


def generator(rows):

    for row in rows:

        try:

            image = load_image(
                row["filename"]
            )

            age = float(
                row["age"]
            )

            race = int(
                row["race"]
            )

            yield (
                image,
                {
                    "age": np.float32(age),
                    "race": np.int32(race)
                }
            )

        except Exception as e:

            print(
                "Image loading error:",
                row["filename"],
                e
            )


# ==================================================
# TF.DATA
# ==================================================

output_signature = (
    tf.TensorSpec(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=tf.float32
    ),
    {
        "age": tf.TensorSpec(
            shape=(),
            dtype=tf.float32
        ),
        "race": tf.TensorSpec(
            shape=(),
            dtype=tf.int32
        )
    }
)

train_ds = tf.data.Dataset.from_generator(
    lambda: generator(train_rows),
    output_signature=output_signature
)

val_ds = tf.data.Dataset.from_generator(
    lambda: generator(val_rows),
    output_signature=output_signature
)

test_ds = tf.data.Dataset.from_generator(
    lambda: generator(test_rows),
    output_signature=output_signature
)

# ==================================================
# DATA AUGMENTATION
# ==================================================

augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        )
    ],
    name="augmentation"
)

# ==================================================
# CNN MODEL
# ==================================================

inputs = keras.Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    name="image"
)

x = augmentation(inputs)

# Block 1
x = layers.Conv2D(
    32,
    3,
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D()(x)

x = layers.Dropout(0.20)(x)

# Block 2
x = layers.Conv2D(
    64,
    3,
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D()(x)

x = layers.Dropout(0.25)(x)

# Block 3
x = layers.Conv2D(
    128,
    3,
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D()(x)

x = layers.Dropout(0.30)(x)

# Block 4
x = layers.Conv2D(
    256,
    3,
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D()(x)

x = layers.Dropout(0.30)(x)

# Shared feature layer
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.40)(x)

# ==================================================
# OUTPUTS
# ==================================================

age_output = layers.Dense(
    1,
    activation="linear",
    name="age"
)(x)

race_output = layers.Dense(
    5,
    activation="softmax",
    name="race"
)(x)

model = keras.Model(
    inputs=inputs,
    outputs={
        "age": age_output,
        "race": race_output
    }
)

# ==================================================
# MODEL SUMMARY
# ==================================================

print("")
print("============================================")
print(" MODEL")
print("============================================")
print("")

model.summary()

# ==================================================
# COMPILE
# ==================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss={
        "age": keras.losses.Huber(),

        "race": keras.losses.SparseCategoricalCrossentropy()
    },

    loss_weights={
        "age": 1.0,
        "race": 1.0
    },

    metrics={
        "age": [
            keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],

        "race": [
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    }
)

# ==================================================
# CALLBACKS
# ==================================================

checkpoint = keras.callbacks.ModelCheckpoint(

    MODEL_FILE,

    monitor="val_race_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)

early_stop = keras.callbacks.EarlyStopping(

    monitor="val_race_accuracy",

    mode="max",

    patience=7,

    restore_best_weights=True,

    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(

    monitor="val_race_accuracy",

    mode="max",

    factor=0.5,

    patience=3,

    min_lr=1e-6,

    verbose=1
)

# ==================================================
# TRAIN
# ==================================================

print("")
print("============================================")
print(" TRAINING STARTED")
print("============================================")
print("")

history = model.fit(

    train_ds.batch(
        BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE
    ),

    validation_data=
        val_ds.batch(
            BATCH_SIZE
        ).prefetch(
            tf.data.AUTOTUNE
        ),

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],

    verbose=1
)

# ==================================================
# SAVE FINAL MODEL
# ==================================================

model.save(
    MODEL_FILE
)

print("")
print("Model saved:")
print(MODEL_FILE)

# ==================================================
# SAVE HISTORY
# ==================================================

history_data = {}

for key, values in history.history.items():

    history_data[key] = [
        float(value)
        for value in values
    ]

with open(
    HISTORY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history_data,
        f,
        indent=2
    )

print("")
print("History saved:")
print(HISTORY_FILE)

# ==================================================
# TEST EVALUATION
# ==================================================

print("")
print("============================================")
print(" TEST EVALUATION")
print("============================================")
print("")

results = model.evaluate(

    test_ds.batch(
        BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE
    ),

    return_dict=True,

    verbose=1
)

print("")
print("============================================")
print(" TEST RESULTS")
print("============================================")
print("")

for key, value in results.items():

    print(
        f"{key}: {value:.4f}"
    )

# ==================================================
# FINAL
# ==================================================

print("")
print("============================================")
print(" TRAINING COMPLETED")
print("============================================")
print("")

