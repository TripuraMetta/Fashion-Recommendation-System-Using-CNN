import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
import streamlit as st
from numpy.linalg import norm

st.header('Fashion Recommendation System')

# Load pickle files
with open('Images_features.pkl', 'rb') as f:
    Image_features = pkl.load(f)
with open('filenames.pkl', 'rb') as f:
    filenames = pkl.load(f)

# Initialize ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Function to extract features from image
def extract_features_from_images(image_path, model):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_expand_dim = np.expand_dims(img_array, axis=0)
        img_preprocess = preprocess_input(img_expand_dim)
        result = model.predict(img_preprocess, verbose=0).flatten()
        norm_result = result / norm(result)
        return norm_result
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Initialize NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# File uploader
upload_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
if upload_file is not None:
    # Create upload directory if it doesn't exist
    os.makedirs('upload', exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join('upload', upload_file.name)
    with open(file_path, 'wb') as f:
        f.write(upload_file.getbuffer())
    
    # Display uploaded image
    st.subheader('Uploaded Image')
    st.image(upload_file)
    
    # Extract features and find similar images
    input_img_features = extract_features_from_images(file_path, model)
    if input_img_features is not None:
        distance, indices = neighbors.kneighbors([input_img_features])
        st.subheader('Recommended Images')
        cols = st.columns(5)
        for i, idx in enumerate(indices[0][1:6]):  # Skip first index (input image)
            if idx < len(filenames):
                with cols[i]:
                    st.image(filenames[idx])
            else:
                with cols[i]:
                    st.write("No more images")