import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Configure Streamlit page
st.set_page_config(page_title='Bamboo Plant Classifier', page_icon='🌱', layout='wide')

# Set up the app title and description
st.title('Bamboo Plant Classifier 🌿')
st.markdown("""
This web app classifies bamboo plant images into different categories.
Upload an image of a bamboo plant to get started with classification.
""")

# Define paths for model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/1class_indices.json"))

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to predict the class of the image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# File uploader
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    # Display original and resized images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Original Image')
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader('Resized Image (224x224)')
        resized_img = image.resize((351, 351))
        st.image(resized_img, use_column_width=True)

    # Classification button
    if st.button('Classify'):
        with st.spinner('Classifying...'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {prediction}')

# Additional information
st.sidebar.header('About')
st.sidebar.info("""
This app uses a pre-trained deep learning model to classify images of bamboo plants.
The model was trained on a dataset of bamboo plant images and can categorize them into different classes.
""")
