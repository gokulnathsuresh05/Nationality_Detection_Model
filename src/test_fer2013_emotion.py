import os
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog

MODEL_PATH = r"F:\Nationality_Detection_Model\models\fer2013_emotion.keras"

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

print("=" * 70)
print("          FER2013 EMOTION MODEL TEST")
print("=" * 70)

try:
    # ---------------------------------------------------------
    # 1. Check model
    # ---------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

    print("\n[1] Loading emotion model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

    # ---------------------------------------------------------
    # 2. Open image picker
    # ---------------------------------------------------------
    print("\n[2] Select an image...")

    root = tk.Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title="Select Image for Emotion Detection",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    if not image_path:
        print("\nNo image selected.")
        input("\nPress ENTER to exit...")
        raise SystemExit

    print(f"\nSelected Image:")
    print(image_path)

    # ---------------------------------------------------------
    # 3. Read image
    # ---------------------------------------------------------
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the selected image.")

    print(f"\nOriginal Image Shape : {image.shape}")

    # ---------------------------------------------------------
    # 4. Convert to grayscale
    # ---------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # FER2013 model expects 48x48 grayscale
    face = cv2.resize(gray, (48, 48))

    # Normalize 0-1
    face = face.astype("float32") / 255.0

    # Shape -> (1, 48, 48, 1)
    input_image = np.expand_dims(face, axis=0)
    input_image = np.expand_dims(input_image, axis=-1)

    print(f"Model Input Shape    : {input_image.shape}")

    # ---------------------------------------------------------
    # 5. Prediction
    # ---------------------------------------------------------
    print("\n[3] Predicting emotion...")

    prediction = model.predict(input_image, verbose=0)

    probabilities = prediction[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_emotion = EMOTIONS[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    # ---------------------------------------------------------
    # 6. Display result
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("                 EMOTION RESULT")
    print("=" * 70)

    print(f"\nPredicted Emotion : {predicted_emotion.upper()}")
    print(f"Confidence        : {confidence:.2f}%")

    print("\nAll Emotion Probabilities:")
    print("-" * 40)

    for emotion, probability in zip(EMOTIONS, probabilities):
        print(f"{emotion:<12}: {probability * 100:6.2f}%")

    print("\n" + "=" * 70)
    print("FER2013 EMOTION TEST COMPLETED")
    print("=" * 70)

except Exception as e:
    print("\n" + "=" * 70)
    print("ERROR")
    print("=" * 70)
    print(type(e).__name__, ":", e)

input("\nPress ENTER to exit...")
