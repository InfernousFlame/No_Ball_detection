import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="No-Ball Detection", layout="centered")

st.title(" No-Ball Detection App")
st.write("Upload a cricket delivery image to predict if it's Legal or No ball")

# Configuration
model_path = './noball_detection_model_quantized.tflite'
img_size = (128, 128)
class_names = ['Legal', 'No ball']

# Try to load model
model_loaded = False
interpreter = None

try:
    import tensorflow as tf
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        model_loaded = True
        st.success(" Model loaded successfully")
except Exception as e:
    st.warning(f" Could not load TensorFlow model: {str(e)}")
    st.info("Running in demo mode - predictions will be simulated")

# File uploader
uploaded_file = st.file_uploader(" Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process image
    img_resized = image.resize(img_size)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    
    st.subheader(" Prediction")
    
    with st.spinner(" Analyzing..."):
        try:
            if model_loaded and interpreter:
                # Use actual model
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                input_data = np.expand_dims(img_array, axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                output_data = interpreter.get_tensor(output_details[0]['index'])
                prediction = output_data[0]
            else:
                # Demo prediction
                prediction = np.array([0.65, 0.35])
            
            # Get prediction
            pred_class = int(np.argmax(prediction))
            confidence = float(prediction[pred_class]) * 100
            label = class_names[pred_class]
            
            # Display result
            col1, col2 = st.columns(2)
            with col1:
                if label == "No ball":
                    st.error(f" **{label.upper()}**")
                else:
                    st.success(f" **{label.upper()}**")
            
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show all probabilities
            st.subheader("Probability Breakdown")
            for i, class_name in enumerate(class_names):
                prob = float(prediction[i]) * 100
                st.write(f"**{class_name}**: {prob:.2f}%")
                st.progress(prediction[i])
        
        except Exception as e:
            st.error(f" Error during prediction: {str(e)}")

st.markdown("---")
st.markdown("""
### About
Cricket No-Ball Detection using Machine Learning
- **Input**: 128x128 RGB image
- **Output**: Legal or No ball classification
- **Model**: Quantized CNN
""")
