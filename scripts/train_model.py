from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# -------------------- APP INIT --------------------
app = Flask(__name__)

# -------------------- CONFIG --------------------
MODEL_PATH = "fan_switch_bulb_model.h5"
UPLOAD_FOLDER = "uploads"
IMG_SIZE = (224, 224)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class_indices = {
    'bulb': 0,
    'fan': 1,
    'charger': 2
}
class_labels = {v: k for k, v in class_indices.items()}

# -------------------- LOAD MODEL --------------------
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# -------------------- HOME (BROWSER) --------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # ---------- FILE CHECK ----------
            if "file" not in request.files:
                return render_template("index.html",
                                       prediction="No file received",
                                       confidence="")

            file = request.files["file"]

            if file.filename == "":
                return render_template("index.html",
                                       prediction="No file selected",
                                       confidence="")

            # ---------- SAVE IMAGE ----------
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            print("Image saved at:", file_path)

            # ---------- LOAD IMAGE ----------
            img = image.load_img(file_path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            print("Image shape:", img_array.shape)

            # ---------- PREDICT ----------
            preds = model.predict(img_array)
            class_index = int(np.argmax(preds))
            label = class_labels[class_index]
            confidence = round(float(np.max(preds)) * 100, 2)

            print("Prediction:", label, confidence)

            return render_template("index.html",
                                   prediction=label,
                                   confidence=confidence)

        except Exception as e:
            print("ERROR:", e)
            return render_template("index.html",
                                   prediction="Error occurred",
                                   confidence="")

    return render_template("index.html")


# -------------------- API (POSTMAN OPTIONAL) --------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        if "file" not in request.files:
            return jsonify({"error": "file not provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "empty filename"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        class_index = int(np.argmax(preds))
        label = class_labels[class_index]
        confidence = round(float(np.max(preds)) * 100, 2)

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)
