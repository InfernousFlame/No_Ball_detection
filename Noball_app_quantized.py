import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# TFLite doesn't require TensorFlow!
try:
    import tensorflow as tf
except ImportError:
    # Fallback: use tflite-runtime for lighter deployment
    try:
        import tflite_runtime.interpreter as tflite
        tf = None
    except:
        st.error("TensorFlow or TFLite runtime not available")

# --- Configuration ---
model_path = './noball_detection_model_quantized.tflite'
img_size = (128, 128)
class_names = ['Legal', 'No ball']

# --- Load Model ---
@st.cache_resource
def load_tflite_model():
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}")
        st.stop()
    
    try:
        # Try using tflite_runtime first (lighter)
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter, True
    except:
        # Fallback to TensorFlow Lite
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter, False

interpreter, is_tflite_runtime = load_tflite_model()

def predict_with_tflite(interpreter, img_array):
    """Make prediction using TFLite interpreter"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input
    input_data = np.expand_dims(img_array, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# --- Streamlit App Layout ---
st.title("No-Ball Detection App")
st.write("Upload a cricket delivery image and the model will predict if it''s a 'Legal' or 'No ball'.")

# --- Image Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    st.write("**Processing image...**")
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    processed_img = cv2.resize(img_array, img_size)
    processed_img = processed_img / 255.0
    processed_img = processed_img.astype(np.float32)

    with st.spinner('Predicting...'):
        try:
            prediction = predict_with_tflite(interpreter, processed_img)
            predicted_class_index = np.argmax(prediction)
            predicted_label = class_names[predicted_class_index]
            confidence = float(prediction[predicted_class_index]) * 100

            st.write("**Prediction Result:**")
            if predicted_label == 'No ball':
                st.error(f"Predicted: **{predicted_label}** (Confidence: {confidence:.2f}%) ")
            else:
                st.success(f"Predicted: **{predicted_label}** (Confidence: {confidence:.2f}%) ")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
