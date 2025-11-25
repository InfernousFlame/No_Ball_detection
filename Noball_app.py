
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# --- Configuration ---
model_path = './noball_detection_model.keras' # Ensure this path is correct relative to where you run Noball_app.py
img_size = (128, 128) # Must match the input size used during training
class_names = ['Legal', 'No ball'] # Must match the class names from training

# --- Load Model ---
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_my_model():
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}. Please check the path.")
        st.stop() # Stop the app if model loading failed
    model = tf.keras.models.load_model(model_path)
    return model

model = load_my_model()

# --- Streamlit App Layout ---
st.title("No-Ball Detection App")
st.write("Upload an image and the model will predict if it's a 'Legal' or 'No ball'.")

if model is None:
    st.stop() # Ensure app stops if model loading failed

# --- Image Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("") # Add a small space

    # Preprocess the image
    st.write("**Processing image...**")
    img_array = np.array(image)
    # Convert RGB to BGR if needed (PIL loads as RGB, OpenCV expects BGR)
    if img_array.shape[2] == 3 and image.mode == 'RGB': 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    processed_img = cv2.resize(img_array, img_size)
    processed_img = processed_img / 255.0  # Normalize
    processed_img = np.expand_dims(processed_img, axis=0) # Add batch dimension

    # Make prediction
    with st.spinner('Predicting...'):
        prediction = model.predict(processed_img)
        predicted_class_index = np.argmax(prediction)
        predicted_label = class_names[predicted_class_index]
        confidence = prediction[0][predicted_class_index] * 100

    st.write("**Prediction Result:**")
    if predicted_label == 'No ball':
        st.error(f"Predicted: **{predicted_label}** (Confidence: {confidence:.2f}%) 🚫")
    else:
        st.success(f"Predicted: **{predicted_label}** (Confidence: {confidence:.2f}%) ✅")

