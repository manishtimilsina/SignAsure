import tensorflow as tf
import cv2
import numpy as np
from django.shortcuts import render
from tensorflow.keras.models import load_model
from django.http import HttpResponse
from .forms import SignatureForm
from .models import Signature
from django.conf import settings
import os
from train_model.train_siamese_model import compute_distance  # Import the custom compute_distance function

# Load the pre-trained Siamese model
model_path = os.path.join(settings.BASE_DIR, 'model/trained_model.h5')
model = tf.keras.models.load_model(model_path, custom_objects={'compute_distance': compute_distance})  # Pass the custom function

# Image preprocessing parameters
IMG_HEIGHT, IMG_WIDTH = 105, 105

# Image preprocessing function
def load_and_preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img.astype("float32") / 255.0
    return img.reshape((1, IMG_HEIGHT, IMG_WIDTH, 1))

# Function to predict similarity between two signatures
def predict_signature(image_a, image_b):
    img_a = load_and_preprocess_image(image_a)
    img_b = load_and_preprocess_image(image_b)
    prediction = model.predict([img_a, img_b])
    return prediction[0][0]

# View to handle signature upload and comparison
def upload_signature(request):
    prediction = None
    file_url = None

    if request.method == 'POST' and request.FILES.get('image_a') and request.FILES.get('image_b'):
        signature_a = request.FILES['image_a']
        signature_b = request.FILES['image_b']

        # Convert images to arrays
        img_a = signature_a.read()
        img_b = signature_b.read()

        img_array_a = np.frombuffer(img_a, np.uint8)
        img_array_b = np.frombuffer(img_b, np.uint8)

        uploaded_image_a = cv2.imdecode(img_array_a, cv2.IMREAD_COLOR)
        uploaded_image_b = cv2.imdecode(img_array_b, cv2.IMREAD_COLOR)

        # Check if both images are valid
        if uploaded_image_a is None or uploaded_image_b is None:
            return render(request, 'compare_signatures.html', {'error': 'Invalid image files uploaded.'})

        # Predict similarity between the two uploaded images
        prediction = predict_signature(uploaded_image_a, uploaded_image_b)

        # Get the URL of the first uploaded image (you can adjust this logic based on how you store the files)
        file_url = signature_a.url if hasattr(signature_a, 'url') else None

    return render(request, 'compare_signatures.html', {'prediction': prediction, 'file_url': file_url})
