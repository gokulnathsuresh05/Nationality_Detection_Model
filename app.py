import os
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

ROOT = r"F:\Nationality_Detection_Model"

AGE_RACE_MODEL = os.path.join(
    ROOT, "models", "utkface_age_race.keras"
)

EMOTION_MODEL = os.path.join(
    ROOT, "models", "fer2013_emotion.keras"
)

CLOTHING_MODEL = os.path.join(
    ROOT, "models", "human_clothing.keras"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

AGE_RACE_SIZE = (128, 128)
EMOTION_SIZE = (48, 48)
CLOTHING_SIZE = (128, 128)

RACE_LABELS = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Other"
}

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

CLOTHING_LABELS = [
    "Black",
    "Blue",
    "Green",
    "Other",
    "Red",
    "White",
    "Yellow"
]


# ============================================================
# CHECK MODEL FILES
# ============================================================

for model_path in [
    AGE_RACE_MODEL,
    EMOTION_MODEL,
    CLOTHING_MODEL
]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Required model not found:\n\n{model_path}"
        )


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print(" NATIONALITY DETECTION MODEL")
print("=" * 70)
print()
print("Loading models...")

age_race_model = tf.keras.models.load_model(
    AGE_RACE_MODEL
)

emotion_model = tf.keras.models.load_model(
    EMOTION_MODEL
)

clothing_model = tf.keras.models.load_model(
    CLOTHING_MODEL
)

print("Age + Ethnicity model loaded.")
print("Emotion model loaded.")
print("Dress Colour model loaded.")
print()
print("All models loaded successfully.")
print()


# ============================================================
# HELPER
# ============================================================

def get_probabilities(values):

    values = np.asarray(
        values,
        dtype=np.float32
    ).flatten()

    if len(values) == 0:
        return values

    total = float(np.sum(values))

    if (
        np.all(values >= 0)
        and np.all(values <= 1)
        and abs(total - 1.0) < 0.05
    ):
        return values

    exp_values = np.exp(
        values - np.max(values)
    )

    return exp_values / np.sum(exp_values)


# ============================================================
# AGE + ETHNICITY PREDICTION
# ============================================================

def predict_age_race(image):

    img = image.convert("RGB")

    img = img.resize(
        AGE_RACE_SIZE
    )

    arr = np.asarray(
        img,
        dtype=np.float32
    ) / 255.0

    arr = np.expand_dims(
        arr,
        axis=0
    )

    prediction = age_race_model.predict(
        arr,
        verbose=0
    )

    age_output = None
    race_output = None

    if isinstance(prediction, dict):

        for key, value in prediction.items():

            arr_value = np.asarray(value)

            if arr_value.ndim >= 1:

                last_dim = arr_value.shape[-1]

                if last_dim == 5:
                    race_output = arr_value

                elif last_dim == 1:
                    age_output = arr_value

    elif isinstance(prediction, (list, tuple)):

        for value in prediction:

            arr_value = np.asarray(value)

            if arr_value.ndim >= 1:

                last_dim = arr_value.shape[-1]

                if last_dim == 5:
                    race_output = arr_value

                elif last_dim == 1:
                    age_output = arr_value

    else:

        arr_value = np.asarray(
            prediction
        )

        if (
            arr_value.ndim >= 2
            and arr_value.shape[-1] == 6
        ):
            age_output = arr_value[..., :1]
            race_output = arr_value[..., 1:]

    if age_output is None or race_output is None:

        raise RuntimeError(
            "Unable to identify Age/Race outputs."
        )

    age = float(
        np.asarray(
            age_output
        ).reshape(-1)[0]
    )

    if 0 <= age <= 1.5:
        age = age * 100.0

    race_values = np.asarray(
        race_output
    ).reshape(-1)

    if len(race_values) != 5:

        raise RuntimeError(
            f"Expected 5 race classes, "
            f"but received {len(race_values)}."
        )

    race_probs = get_probabilities(
        race_values
    )

    race_index = int(
        np.argmax(race_probs)
    )

    race_confidence = (
        float(
            race_probs[race_index]
        ) * 100.0
    )

    return (
        age,
        race_index,
        race_confidence
    )


# ============================================================
# EMOTION PREDICTION
# ============================================================

def predict_emotion(image):

    img = image.convert("L")

    img = img.resize(
        EMOTION_SIZE
    )

    arr = np.asarray(
        img,
        dtype=np.float32
    ) / 255.0

    arr = np.expand_dims(
        arr,
        axis=-1
    )

    arr = np.expand_dims(
        arr,
        axis=0
    )

    prediction = emotion_model.predict(
        arr,
        verbose=0
    )

    values = np.asarray(
        prediction
    ).reshape(-1)

    if len(values) != 7:

        raise RuntimeError(
            f"Expected 7 emotion classes, "
            f"but received {len(values)}."
        )

    probabilities = get_probabilities(
        values
    )

    index = int(
        np.argmax(probabilities)
    )

    confidence = (
        float(
            probabilities[index]
        ) * 100.0
    )

    return (
        EMOTION_LABELS[index],
        confidence
    )


# ============================================================
# CLOTHING PREDICTION
# ============================================================

def predict_clothing(image):

    img = image.convert("RGB")

    img = img.resize(
        CLOTHING_SIZE
    )

    arr = np.asarray(
        img,
        dtype=np.float32
    ) / 255.0

    arr = np.expand_dims(
        arr,
        axis=0
    )

    prediction = clothing_model.predict(
        arr,
        verbose=0
    )

    values = np.asarray(
        prediction
    ).reshape(-1)

    if len(values) != 7:

        raise RuntimeError(
            f"Expected 7 clothing classes, "
            f"but received {len(values)}."
        )

    probabilities = get_probabilities(
        values
    )

    index = int(
        np.argmax(probabilities)
    )

    confidence = (
        float(
            probabilities[index]
        ) * 100.0
    )

    return (
        CLOTHING_LABELS[index],
        confidence
    )


# ============================================================
# GUI CLASS
# ============================================================

class NationalityDetectionGUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Nationality Detection Model"
        )

        self.root.geometry(
            "1150x800"
        )

        self.root.minsize(
            1000,
            720
        )

        self.root.configure(
            bg="#F4F6F8"
        )

        self.selected_image_path = None
        self.original_image = None
        self.preview_photo = None

        self.create_gui()


    # ========================================================
    # CREATE GUI
    # ========================================================

    def create_gui(self):

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = tk.Frame(
            self.root,
            bg="#17202A",
            height=90
        )

        header.pack(
            fill="x"
        )

        header.pack_propagate(
            False
        )

        title = tk.Label(
            header,
            text="Nationality Detection Model",
            font=(
                "Segoe UI",
                25,
                "bold"
            ),
            bg="#17202A",
            fg="white"
        )

        title.pack(
            pady=(14, 0)
        )

        subtitle = tk.Label(
            header,
            text="AI-Based Ethnicity / Group, Age, Emotion & Dress Colour Analysis",
            font=(
                "Segoe UI",
                11
            ),
            bg="#17202A",
            fg="#D5D8DC"
        )

        subtitle.pack()


        # ----------------------------------------------------
        # BUTTONS AT TOP
        # ----------------------------------------------------

        button_area = tk.Frame(
            self.root,
            bg="#F4F6F8"
        )

        button_area.pack(
            pady=(20, 10)
        )

        self.select_button = tk.Button(
            button_area,
            text="SELECT IMAGE",
            command=self.select_image,
            font=(
                "Segoe UI",
                11,
                "bold"
            ),
            bg="#2874A6",
            fg="white",
            activebackground="#1F618D",
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=35,
            pady=14,
            cursor="hand2"
        )

        self.select_button.grid(
            row=0,
            column=0,
            padx=8
        )


        self.predict_button = tk.Button(
            button_area,
            text="PREDICT",
            command=self.predict,
            font=(
                "Segoe UI",
                11,
                "bold"
            ),
            bg="#229954",
            fg="white",
            activebackground="#1E8449",
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=45,
            pady=14,
            cursor="hand2",
            state="disabled"
        )

        self.predict_button.grid(
            row=0,
            column=1,
            padx=8
        )


        self.clear_button = tk.Button(
            button_area,
            text="CLEAR",
            command=self.clear,
            font=(
                "Segoe UI",
                11,
                "bold"
            ),
            bg="#CB4335",
            fg="white",
            activebackground="#A93226",
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=45,
            pady=14,
            cursor="hand2"
        )

        self.clear_button.grid(
            row=0,
            column=2,
            padx=8
        )


        # ----------------------------------------------------
        # MAIN AREA
        # ----------------------------------------------------

        main = tk.Frame(
            self.root,
            bg="#F4F6F8"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=10
        )


        # ====================================================
        # LEFT IMAGE PANEL
        # ====================================================

        left = tk.Frame(
            main,
            bg="white",
            highlightthickness=1,
            highlightbackground="#D5D8DC"
        )

        left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 12)
        )


        preview_title = tk.Label(
            left,
            text="IMAGE PREVIEW",
            font=(
                "Segoe UI",
                14,
                "bold"
            ),
            bg="white",
            fg="#17202A"
        )

        preview_title.pack(
            pady=(15, 10)
        )


        # Medium-large preview area

        self.preview_frame = tk.Frame(
            left,
            bg="#EEF1F4",
            width=500,
            height=500
        )

        self.preview_frame.pack(
            padx=20,
            pady=10,
            fill="both",
            expand=True
        )

        self.preview_frame.pack_propagate(
            False
        )


        self.preview_label = tk.Label(
            self.preview_frame,
            text="No image selected\n\nClick SELECT IMAGE",
            font=(
                "Segoe UI",
                14
            ),
            bg="#EEF1F4",
            fg="#7B7D7D",
            justify="center"
        )

        self.preview_label.pack(
            fill="both",
            expand=True
        )


        # ====================================================
        # RIGHT RESULT PANEL
        # ====================================================

        right = tk.Frame(
            main,
            bg="white",
            highlightthickness=1,
            highlightbackground="#D5D8DC"
        )

        right.pack(
            side="right",
            fill="both",
            expand=True,
            padx=(12, 0)
        )


        result_title = tk.Label(
            right,
            text="PREDICTION RESULT",
            font=(
                "Segoe UI",
                14,
                "bold"
            ),
            bg="white",
            fg="#17202A"
        )

        result_title.pack(
            pady=(15, 10)
        )


        self.result_box = tk.Frame(
            right,
            bg="#F8F9F9"
        )

        self.result_box.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )


        self.result_text = tk.Label(
            self.result_box,
            text="Prediction results will appear here.",
            font=(
                "Segoe UI",
                13
            ),
            bg="#F8F9F9",
            fg="#566573",
            justify="left",
            anchor="nw"
        )

        self.result_text.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=25
        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        self.status_label = tk.Label(
            self.root,
            text="Ready - Please select an image.",
            font=(
                "Segoe UI",
                10
            ),
            bg="#F4F6F8",
            fg="#566573"
        )

        self.status_label.pack(
            pady=(5, 15)
        )


    # ========================================================
    # SELECT IMAGE
    # ========================================================

    def select_image(self):

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                ),
                (
                    "JPEG Files",
                    "*.jpg *.jpeg"
                ),
                (
                    "PNG Files",
                    "*.png"
                ),
                (
                    "All Files",
                    "*.*"
                )
            ]
        )

        if not file_path:
            return

        try:

            image = Image.open(
                file_path
            ).convert("RGB")

            self.selected_image_path = file_path
            self.original_image = image

            self.show_preview(
                image
            )

            self.result_text.config(
                text=(
                    "Image selected successfully.\n\n"
                    "Click PREDICT to analyze."
                ),
                fg="#566573"
            )

            self.status_label.config(
                text=(
                    "Selected: "
                    + os.path.basename(
                        file_path
                    )
                ),
                fg="#2874A6"
            )

            self.predict_button.config(
                state="normal"
            )

        except Exception as e:

            messagebox.showerror(
                "Image Error",
                f"Unable to open image.\n\n{e}"
            )


    # ========================================================
    # SHOW MEDIUM IMAGE
    # ========================================================

    def show_preview(self, image):

        preview = image.copy()

        # Medium-large display size
        max_width = 470
        max_height = 470

        preview.thumbnail(
            (
                max_width,
                max_height
            ),
            Image.Resampling.LANCZOS
        )

        self.preview_photo = ImageTk.PhotoImage(
            preview
        )

        self.preview_label.config(
            image=self.preview_photo,
            text=""
        )


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self):

        if self.original_image is None:

            messagebox.showwarning(
                "No Image",
                "Please select an image first."
            )

            return

        try:

            self.status_label.config(
                text="Analyzing image... Please wait.",
                fg="#D68910"
            )

            self.root.update_idletasks()


            # ------------------------------------------------
            # AGE + GROUP
            # ------------------------------------------------

            age, race_index, race_confidence = (
                predict_age_race(
                    self.original_image
                )
            )

            race = RACE_LABELS.get(
                race_index,
                "Other"
            )


            # ------------------------------------------------
            # EMOTION
            # ------------------------------------------------

            emotion, emotion_confidence = (
                predict_emotion(
                    self.original_image
                )
            )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            result_lines = []

            result_lines.append(
                "MODEL-BASED CLASSIFICATION"
            )

            result_lines.append(
                "--------------------------------"
            )

            result_lines.append(
                f"Ethnicity / Group : {race}"
            )

            result_lines.append(
                f"Group Confidence  : {race_confidence:.2f}%"
            )

            result_lines.append("")


            # ------------------------------------------------
            # INDIAN
            # ------------------------------------------------

            if race_index == 3:

                result_lines.append(
                    "ASSIGNMENT GROUP : INDIAN"
                )

                result_lines.append("")

                result_lines.append(
                    f"Age               : {age:.0f} years"
                )

                result_lines.append(
                    f"Emotion           : {emotion}"
                )

                result_lines.append(
                    f"Emotion Confidence: "
                    f"{emotion_confidence:.2f}%"
                )


                clothing, clothing_confidence = (
                    predict_clothing(
                        self.original_image
                    )
                )

                result_lines.append(
                    f"Dress Colour      : {clothing}"
                )

                result_lines.append(
                    f"Dress Confidence  : "
                    f"{clothing_confidence:.2f}%"
                )


            # ------------------------------------------------
            # WHITE / UNITED STATES
            # ------------------------------------------------

            elif race_index == 0:

                result_lines.append(
                    "ASSIGNMENT GROUP : "
                    "UNITED STATES / WHITE"
                )

                result_lines.append("")

                result_lines.append(
                    f"Age               : {age:.0f} years"
                )

                result_lines.append(
                    f"Emotion           : {emotion}"
                )

                result_lines.append(
                    f"Emotion Confidence: "
                    f"{emotion_confidence:.2f}%"
                )


            # ------------------------------------------------
            # BLACK / AFRICAN
            # ------------------------------------------------

            elif race_index == 1:

                result_lines.append(
                    "ASSIGNMENT GROUP : "
                    "AFRICAN / BLACK"
                )

                result_lines.append("")

                result_lines.append(
                    f"Emotion           : {emotion}"
                )

                result_lines.append(
                    f"Emotion Confidence: "
                    f"{emotion_confidence:.2f}%"
                )


                clothing, clothing_confidence = (
                    predict_clothing(
                        self.original_image
                    )
                )

                result_lines.append(
                    f"Dress Colour      : {clothing}"
                )

                result_lines.append(
                    f"Dress Confidence  : "
                    f"{clothing_confidence:.2f}%"
                )


            # ------------------------------------------------
            # ASIAN / OTHER
            # ------------------------------------------------

            else:

                if race_index == 2:
                    assignment_group = "ASIAN"
                else:
                    assignment_group = "OTHER"

                result_lines.append(
                    f"ASSIGNMENT GROUP : "
                    f"{assignment_group}"
                )

                result_lines.append("")

                result_lines.append(
                    f"Ethnicity / Group : {race}"
                )

                result_lines.append(
                    f"Emotion           : {emotion}"
                )

                result_lines.append(
                    f"Emotion Confidence: "
                    f"{emotion_confidence:.2f}%"
                )


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            self.result_text.config(
                text="\n".join(
                    result_lines
                ),
                fg="#17202A",
                font=(
                    "Segoe UI",
                    12
                )
            )

            self.status_label.config(
                text="Prediction completed successfully.",
                fg="#229954"
            )


        except Exception as e:

            self.status_label.config(
                text="Prediction failed.",
                fg="#CB4335"
            )

            print()
            print("=" * 70)
            print("PREDICTION ERROR")
            print("=" * 70)
            traceback.print_exc()
            print("=" * 70)
            print()

            messagebox.showerror(
                "Prediction Error",
                str(e)
            )


    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self):

        self.selected_image_path = None
        self.original_image = None
        self.preview_photo = None


        self.preview_label.config(
            image="",
            text=(
                "No image selected\n\n"
                "Click SELECT IMAGE"
            ),
            font=(
                "Segoe UI",
                14
            ),
            fg="#7B7D7D"
        )


        self.result_text.config(
            text="Prediction results will appear here.",
            font=(
                "Segoe UI",
                13
            ),
            fg="#566573"
        )


        self.status_label.config(
            text="Ready - Please select an image.",
            fg="#566573"
        )


        self.predict_button.config(
            state="disabled"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    root = tk.Tk()

    app = NationalityDetectionGUI(
        root
    )

    root.mainloop()


if __name__ == "__main__":

    try:

        main()

    except Exception:

        print()
        print("=" * 70)
        print("APPLICATION ERROR")
        print("=" * 70)

        traceback.print_exc()

        print("=" * 70)
        print()

        input(
            "Press ENTER to close..."
        )

