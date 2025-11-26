import streamlit as st
import numpy as np
from PIL import Image
import os

# --- Configuration ---
model_path = './noball_detection_model_quantized.tflite'
img_size = (128, 128)
class_names = ['Legal', 'No ball']

# --- Load TFLite model ---
@st.cache_resource
def load_model():
    """Load TFLite model"""
    if not os.path.exists(model_path):
        st.error(f" Model file not found: {model_path}")
        return None
    
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return ('tensorflow', interpreter)
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return ('tflite_runtime', interpreter)
        except ImportError:
            st.warning(" TensorFlow not available - using demo mode")
            return ('demo', None)

def predict(interpreter, img_array, mode):
    """Make prediction"""
    if mode == 'demo':
        # Demo prediction
        return np.array([0.7, 0.3])
    
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_data = np.expand_dims(img_array, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# --- UI ---
st.set_page_config(page_title="No-Ball Detection", layout="centered")
st.title(" No-Ball Detection")
st.write("Upload a cricket delivery image to predict Legal or No ball")

# Load model
model_info = load_model()

# File uploader
uploaded_file = st.file_uploader(" Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.info(" Processing...")
        
        # Resize image
        img_resized = image.resize(img_size)
        img_array = np.array(img_resized) / 255.0
        img_array = img_array.astype(np.float32)
        
        # Predict
        with st.spinner(" Analyzing..."):
            if model_info:
                mode, interpreter = model_info
                prediction = predict(interpreter, img_array, mode)
                
                if prediction is not None:
                    pred_class = np.argmax(prediction)
                    confidence = float(prediction[pred_class]) * 100
                    label = class_names[pred_class]
                    
                    st.markdown("---")
                    
                    if label == "No ball":
                        st.error(f" **NO BALL** ({confidence:.1f}%)")
                    else:
                        st.success(f" **LEGAL** ({confidence:.1f}%)")
                    
                    with st.expander("View Details"):
                        for i, name in enumerate(class_names):
                            prob = prediction[i] * 100
                            st.write(f"{name}: {prob:.2f}%")
            else:
                st.error("Model not available")

st.markdown("---")
st.caption(" Cricket No-Ball Detection ML Model")
