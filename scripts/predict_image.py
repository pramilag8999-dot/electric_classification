# predict_image.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tkinter import Tk, filedialog, messagebox

# 🔹 Load trained model
model = load_model('fan_switch_bulb_model.h5')

IMG_SIZE = (224, 224)

# 🔹 Map class indices (training प्रमाणेच असायला हवेत)
class_indices = {
    'bulb': 0,
    'fan': 1,
    'switch': 2,
    'charger': 3
}

# 🔹 Reverse mapping: index → label
class_labels = {v: k for k, v in class_indices.items()}

# 🔹 GUI to select image
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select Image for Prediction")

if file_path:
    # 🔹 Load & preprocess image
    img = image.load_img(file_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 🔥 PREDICTION
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)

    label = class_labels[idx]
    confidence = prediction[0][idx] * 100

    # 🔹 Show result with confidence
    messagebox.showinfo(
        "Prediction Result",
        f"This image is classified as: {label}\nConfidence: {confidence:.2f}%"
    )
