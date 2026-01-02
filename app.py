from flask import Flask, render_template, request
import numpy as np
import os
import onnxruntime as ort
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Class labels
class_indices = {
    0: 'BULB',
    1: 'CHARGER',
    2: 'FANS'
}

# Load ONNX model
model_path = "model/electric-gadget.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

IMG_SIZE = (224, 224)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class = None
    confidence = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('index.html', error="No file selected")

        # Save uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        try:
            # Preprocess image
            img = Image.open(image_path).convert("RGB")
            img = img.resize(IMG_SIZE)

            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # ONNX inference
            pred = session.run(None, {input_name: img_array})[0][0]

            class_id = int(np.argmax(pred))
            confidence = round(float(pred[class_id]) * 100, 2)
            predicted_class = class_indices.get(class_id, "OTHER")

        except Exception as e:
            print("Prediction error:", e)
            predicted_class = "OTHER"
            confidence = 0.0

    return render_template(
        'index.html',
        predicted_class=predicted_class,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
