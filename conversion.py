import tensorflow as tf
import tf2onnx

file_name = "fan_switch_bulb_model.h5"

model = tf.keras.models.load_model(
    file_name,
    compile=False,
    safe_mode=False
)

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path="fan_switch_bulb_model.onnx"
)

print("success")
