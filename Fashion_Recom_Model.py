import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
from tqdm import tqdm  # For progress bar
import logging  # For logging progress

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Extract Filenames from Folder
filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))
logger.info(f"Found {len(filenames)} images in the dataset")

# Importing ResNet50 Model and Configuration
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])
model.summary()
logger.info("ResNet50 model loaded and configured")

# Function to Extract Features from Image
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
        logger.error(f"Error processing {image_path}: {e}")
        return None

# Extract features for all images with progress bar
image_features = []
logger.info("Starting feature extraction for all images")
for file in tqdm(filenames, desc="Extracting Features", unit="image"):
    features = extract_features_from_images(file, model)
    if features is not None:
        image_features.append(features)
    else:
        logger.warning(f"Skipping {file} due to processing error")

# Save features and filenames to pickle files
logger.info("Saving features and filenames to pickle files")
with open('Images_features.pkl', 'wb') as f:
    pkl.dump(image_features, f)
with open('filenames.pkl', 'wb') as f:
    pkl.dump(filenames, f)

# Load pickle files for verification
logger.info("Loading pickle files for verification")
with open('Images_features.pkl', 'rb') as f:
    Image_features = pkl.load(f)
with open('filenames.pkl', 'rb') as f:
    loaded_filenames = pkl.load(f)

logger.info(f"Processed {len(Image_features)} images")
logger.info(f"Feature shape: {np.array(Image_features).shape}")

# Finding Similar Images
logger.info("Fitting NearestNeighbors model")
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Test with an input image
logger.info("Testing with input image '16871.jpg'")
input_image = extract_features_from_images('16871.jpg', model)
if input_image is not None:
    distance, indices = neighbors.kneighbors([input_image])
    logger.info(f"Recommended image indices: {indices}")

    # Display images (for Jupyter or IPython)
    from IPython.display import Image
    logger.info("Displaying input image")
    Image('16871.jpg')
    for idx in indices[0]:
        logger.info(f"Displaying recommended image: {filenames[idx]}")
        Image(filenames[idx])
else:
    logger.error("Failed to process test image '16871.jpg'")