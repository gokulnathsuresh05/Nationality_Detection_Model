import os
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog

MODEL_PATH = r"F:\Nationality_Detection_Model\models\utkface_age_race.keras"
IMG_SIZE = 128

RACE_LABELS = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Other"
}

print()
print("=" * 60)
print("       UTKFACE AGE + RACE MODEL TEST")
print("=" * 60)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\n[1] Loading trained model...")

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found!")
    print(MODEL_PATH)
    input("\nPress ENTER to exit...")
    raise SystemExit

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print("\nERROR while loading model:")
    print(e)
    input("\nPress ENTER to exit...")
    raise SystemExit

# ------------------------------------------------------------
# Open Windows File Picker
# ------------------------------------------------------------

print()
print("=" * 60)
print("IMAGE SELECTION")
print("=" * 60)

print("\nOpening image selection window...")

try:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    image_path = filedialog.askopenfilename(
        title="Select a Person Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("JPG Files", "*.jpg *.jpeg"),
            ("PNG Files", "*.png"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

except Exception as e:
    print("\nERROR opening image selection window:")
    print(e)
    input("\nPress ENTER to exit...")
    raise SystemExit

if not image_path:
    print("\nNo image selected.")
    input("\nPress ENTER to exit...")
    raise SystemExit

print("\nSelected image:")
print(image_path)

# ------------------------------------------------------------
# Read image
# ------------------------------------------------------------

print("\n[2] Reading image...")

image = cv2.imread(image_path)

if image is None:
    print("\nERROR: Could not read image.")
    input("\nPress ENTER to exit...")
    raise SystemExit

print("Image loaded successfully!")
print("Original image shape:", image.shape)

# ------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------

print("\n[3] Preprocessing image...")

try:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_normalized = image_resized.astype(np.float32) / 255.0
    input_image = np.expand_dims(image_normalized, axis=0)

    print("Input shape:", input_image.shape)

except Exception as e:
    print("\nERROR during preprocessing:")
    print(e)
    input("\nPress ENTER to exit...")
    raise SystemExit

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

print("\n[4] Running prediction...")

try:
    prediction = model.predict(input_image, verbose=0)

except Exception as e:
    print("\nERROR during prediction:")
    print(e)
    input("\nPress ENTER to exit...")
    raise SystemExit

# ------------------------------------------------------------
# Read dictionary output
# ------------------------------------------------------------

try:
    age_output = prediction["age"]
    race_output = prediction["race"]

    predicted_age = float(age_output[0][0])
    race_probabilities = race_output[0]

except Exception as e:
    print("\nERROR reading model output:")
    print(e)

    print("\nRaw prediction:")
    print(prediction)

    input("\nPress ENTER to exit...")
    raise SystemExit

# ------------------------------------------------------------
# Race prediction
# ------------------------------------------------------------

race_index = int(np.argmax(race_probabilities))
predicted_race = RACE_LABELS[race_index]

race_confidence = float(
    race_probabilities[race_index] * 100
)

# ------------------------------------------------------------
# Senior citizen
# ------------------------------------------------------------

if predicted_age > 60:
    senior_status = "YES - Senior Citizen"
else:
    senior_status = "NO - Not Senior Citizen"

# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print()
print("=" * 60)
print("                 PREDICTION RESULT")
print("=" * 60)

print("\nImage:")
print(image_path)

print("\n--------------------------------------------")
print("Age              :", round(predicted_age, 1), "years")
print("Race / Ethnicity :", predicted_race)
print("Confidence       :", round(race_confidence, 2), "%")
print("Senior Citizen   :", senior_status)
print("--------------------------------------------")

# ------------------------------------------------------------
# All probabilities
# ------------------------------------------------------------

print("\nRace probabilities:")

for index, probability in enumerate(race_probabilities):
    label = RACE_LABELS[index]
    percentage = float(probability * 100)

    print(f"  {label:<10} : {percentage:>6.2f}%")

print()
print("=" * 60)
print("                 TEST COMPLETED")
print("=" * 60)

input("\nPress ENTER to exit...")
