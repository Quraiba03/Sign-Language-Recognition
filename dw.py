import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model and labels
model = load_model("C:/Users/Muniq/OneDrive/Desktop/sign_language_detection_main/Model/keras_model.h5")
with open("C:/Users/Muniq/OneDrive/Desktop/sign_language_detection_main/Model/labels.txt", 'r') as file:
    labels = file.readlines()
labels = [label.strip() for label in labels]

# Test with a sample image
img = cv2.imread('path_to_sample_image.jpg')
img_resized = cv2.resize(img, (224, 224))  # Resize to model input size
img_resized = img_resized.astype('float32') / 255.0  # Normalize
img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension

predictions = model.predict(img_resized)
predicted_class = np.argmax(predictions)
print(f"Predicted Class: {labels[predicted_class]}")
