# app/ai_model.py
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os

# --- Configuration ---
MODEL_PATH = "../models/skin_disease_model.h5"
CLASS_NAMES_VERBOSE = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis',
                       'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesion']
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 30.0  # Adjusted based on testing
SKIN_COLOR_THRESHOLD = 0.05
# --- End Configuration ---

# --- Model Loading ---
_model = None


def load_model_once():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        print(f"Loading model from {MODEL_PATH}...")
        _model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    return _model


# --- Heuristic Filter Function ---
def is_skin_color_heuristic(image, skin_color_threshold=SKIN_COLOR_THRESHOLD):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_skin = np.array([0, 48, 80])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
        total_pixels = image.shape[0] * image.shape[1]
        skin_pixels = cv2.countNonZero(skin_mask)
        skin_percentage = skin_pixels / total_pixels
        print(f"Skin percentage heuristic: {skin_percentage:.2f}")
        return skin_percentage >= skin_color_threshold
    except Exception as e:
        print(f"Error in skin color heuristic: {e}")
        return False


# --- Image Preprocessing ---
def preprocess_image_for_prediction(image, target_size=IMAGE_SIZE):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.resize(image, target_size)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


# --- Prediction Logic ---
def predict_image(image):
    """
    Predict the class of a single skin lesion image.
    Includes filtering for skin color and confidence threshold.
    Returns analysis result and recommendation.
    """
    # Pre-filtering
    if not is_skin_color_heuristic(image):
        return {
            'predicted_class': 'Non-Skin Image',
            'predicted_class_code': 'non_skin',
            'confidence': 0.0,
            'all_predictions': {name: 0.0 for name in CLASS_NAMES_VERBOSE},
            'error': 'Image does not appear to contain significant skin-colored pixels.',
            'recommendation': 'Cannot analyze non-skin image.',
            'is_valid_prediction': False,
            'needs_appointment': False
        }

    try:
        model = load_model_once()
        processed_image = preprocess_image_for_prediction(image)
        if processed_image is None:
            return {'error': 'Failed to preprocess image.', 'is_valid_prediction': False, 'needs_appointment': False}

        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx] * 100)  # Convert to Python float
        predicted_class_verbose = CLASS_NAMES_VERBOSE[predicted_class_idx]

        # Convert all predictions to Python floats for JSON serialization
        all_predictions = {CLASS_NAMES_VERBOSE[i]: float(predictions[0][i] * 100) for i in
                           range(len(CLASS_NAMES_VERBOSE))}

        # Confidence check
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                'predicted_class': 'Unclassified',
                'predicted_class_code': 'unclassified',
                'confidence': confidence,
                'all_predictions': all_predictions,  # Already converted to floats
                'error': f'Confidence ({confidence:.2f}%) is below the threshold ({CONFIDENCE_THRESHOLD}%).',
                'recommendation': 'Confidence is low. Consider professional consultation.',
                'is_valid_prediction': False,
                'needs_appointment': True  # Low confidence might warrant a visit
            }

        # Determine recommendation based on prediction and confidence
        needs_appointment = False
        recommendation = f"Confidence is {confidence:.2f}% for {predicted_class_verbose}. "

        if predicted_class_verbose == "Melanoma":
            recommendation += "HIGH RISK - MELANOMA DETECTED. Immediate medical attention is recommended."
            needs_appointment = True
        elif confidence < 60:  # Adjust this threshold for 'moderate confidence'
            recommendation += "Moderate confidence. Consult a dermatologist for confirmation."
            needs_appointment = True
        elif confidence < 80:  # Adjust this threshold for 'high confidence'
            recommendation += "Consider consulting a dermatologist for confirmation."
            needs_appointment = True
        else:
            recommendation += "High confidence prediction. Still, consider a professional opinion for any concerning lesions."

        return {
            'predicted_class': predicted_class_verbose,
            'predicted_class_code': CLASS_NAMES_VERBOSE.index(predicted_class_verbose),
            # Or use the original code if available
            'confidence': confidence,  # Already a Python float
            'all_predictions': all_predictions,  # Already converted to floats
            'error': None,
            'recommendation': recommendation,
            'is_valid_prediction': True,
            'needs_appointment': needs_appointment
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {'error': f"An error occurred during prediction: {str(e)}", 'is_valid_prediction': False,
                'needs_appointment': False}

# --- End Prediction Logic ---