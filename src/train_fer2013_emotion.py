import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# FER2013 EMOTION MODEL - TRAINING
# ============================================================

PROJECT = r"F:\Nationality_Detection_Model"
DATASET = os.path.join(PROJECT, "dataset", "FER2013")

TRAIN_DIR = os.path.join(DATASET, "train")
TEST_DIR = os.path.join(DATASET, "test")

MODEL_DIR = os.path.join(PROJECT, "models")
RESULTS_DIR = os.path.join(PROJECT, "results")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fer2013_emotion.keras"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "fer2013_training_history.json"
)

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30
SEED = 42

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

print()
print("=" * 70)
print("             FER2013 EMOTION MODEL TRAINING")
print("=" * 70)

# ============================================================
# CHECK DATASET
# ============================================================

print("\n[1] Checking dataset...")

if not os.path.exists(TRAIN_DIR):
    print("\nERROR: Train folder not found!")
    print(TRAIN_DIR)
    input("\nPress ENTER to exit...")
    raise SystemExit

if not os.path.exists(TEST_DIR):
    print("\nERROR: Test folder not found!")
    print(TEST_DIR)
    input("\nPress ENTER to exit...")
    raise SystemExit

print("Train:", TRAIN_DIR)
print("Test :", TEST_DIR)

print("\nDataset class counts:")

for emotion in CLASS_NAMES:

    train_path = os.path.join(TRAIN_DIR, emotion)
    test_path = os.path.join(TEST_DIR, emotion)

    train_count = 0
    test_count = 0

    if os.path.exists(train_path):
        train_count = len([
            f for f in os.listdir(train_path)
            if os.path.isfile(os.path.join(train_path, f))
        ])

    if os.path.exists(test_path):
        test_count = len([
            f for f in os.listdir(test_path)
            if os.path.isfile(os.path.join(test_path, f))
        ])

    print(
        f"  {emotion:<10} "
        f"Train: {train_count:>5}   "
        f"Test: {test_count:>5}"
    )

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\n[2] Loading training dataset...")

try:

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.10,
        subset="training"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.10,
        subset="validation"
    )

except Exception as e:

    print("\nERROR loading training dataset:")
    print(e)

    input("\nPress ENTER to exit...")
    raise SystemExit

print("Training dataset loaded.")
print("Validation dataset loaded.")

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n[3] Loading test dataset...")

try:

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

except Exception as e:

    print("\nERROR loading test dataset:")
    print(e)

    input("\nPress ENTER to exit...")
    raise SystemExit

print("Test dataset loaded.")

# ============================================================
# NORMALIZATION
# ============================================================

print("\n[4] Preparing image pipeline...")

AUTOTUNE = tf.data.AUTOTUNE

normalization = layers.Rescaling(1.0 / 255.0)

train_ds = train_ds.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=AUTOTUNE
)

val_ds = val_ds.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=AUTOTUNE
)

test_ds = test_ds.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("Pipeline ready.")

# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10)
    ],
    name="data_augmentation"
)

# ============================================================
# CNN MODEL
# ============================================================

print("\n[5] Building CNN model...")

model = models.Sequential(
    [
        layers.Input(shape=(48, 48, 1)),

        augmentation,

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        layers.Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.35),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.40),

        layers.Dense(
            7,
            activation="softmax"
        )
    ],
    name="FER2013_Emotion_CNN"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

print("\n[6] Preparing callbacks...")

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.000001,
        verbose=1
    )
]

# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("                    TRAINING STARTED")
print("=" * 70)

print("\nEpochs     :", EPOCHS)
print("Batch size :", BATCH_SIZE)
print("Image size : 48 x 48")
print("Channels   : Grayscale")
print("Classes    : 7")

print("\nCPU training may take some time.")
print("Please do NOT close the terminal.\n")

try:

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

except Exception as e:

    print("\nERROR DURING TRAINING:")
    print(e)

    input("\nPress ENTER to exit...")
    raise SystemExit

# ============================================================
# TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("                    TEST EVALUATION")
print("=" * 70)

try:

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1
    )

    print()
    print("Test Loss     :", round(float(test_loss), 4))
    print(
        "Test Accuracy :",
        round(float(test_accuracy) * 100, 2),
        "%"
    )

except Exception as e:

    print("\nERROR during testing:")
    print(e)

    test_loss = None
    test_accuracy = None

# ============================================================
# SAVE MODEL
# ============================================================

print("\n[7] Saving emotion model...")

try:

    model.save(MODEL_PATH)

    print("Model saved successfully!")
    print(MODEL_PATH)

except Exception as e:

    print("\nERROR saving model:")
    print(e)

# ============================================================
# SAVE HISTORY
# ============================================================

print("\n[8] Saving training history...")

try:

    history_data = {}

    for key, values in history.history.items():

        history_data[key] = [
            float(value)
            for value in values
        ]

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=2
        )

    print("Training history saved!")
    print(HISTORY_PATH)

except Exception as e:

    print("\nERROR saving history:")
    print(e)

# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("             FER2013 TRAINING COMPLETED")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

if test_accuracy is not None:

    print(
        "\nFinal Test Accuracy:",
        round(float(test_accuracy) * 100, 2),
        "%"
    )

print("\nEmotion mapping:")

for i, emotion in enumerate(CLASS_NAMES):

    print(
        f"  {i} = {emotion}"
    )

print()
print("=" * 70)

input("\nPress ENTER to exit...")
