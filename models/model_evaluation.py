"""
Model Evaluation Module for Skin Disease Detection System (HAM10000 - 7 Classes)
CIT 4299: IT Project
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import sys

# Add the 'preprocessing' directory to the Python path for PyCharm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))

# Import the required function
from data_preprocessing import create_data_generators

def evaluate_model(model_path, test_generator, class_names):
    """
    Evaluate the trained model
    
    Args:
        model_path (str): Path to the trained model
        test_generator: Test data generator
        class_names (list): List of class names
    
    Returns:
        tuple: (classification_report_dict, confusion_matrix_array)
    """
    # Load model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    # Get predictions
    print("Making predictions...")
    test_steps = test_generator.samples // test_generator.batch_size + 1
    predictions = model.predict(test_generator, steps=test_steps, verbose=1)
    
    # Get predicted classes
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true classes
    true_classes = test_generator.classes
    
    # Generate classification report
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=class_names,
        output_dict=True
    )
    
    # Print classification report
    print("\nClassification Report:")
    print("=" * 50)
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('../documentation/confusion_matrix.png')
    plt.show()
    
    return report, cm

def predict_single_image(model_path, image_path, class_names):
    """
    Predict a single image
    
    Args:
        model_path (str): Path to the trained model
        image_path (str): Path to the image file
        class_names (list): List of class names
    
    Returns:
        dict: Prediction results
    """
    from data_preprocessing import preprocess_image
    
    # Load model
    model = load_model(model_path)
    
    # Preprocess image
    image = preprocess_image(image_path)
    if image is None:
        return None
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Get class name
    predicted_class = class_names[predicted_class_idx]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_predictions': dict(zip(class_names, predictions[0]))
    }

def main():
    """
    Main function to evaluate the trained model using HAM10000 dataset
    """
    print("Starting Model Evaluation (HAM10000 - 7 classes)")
    print("=" * 30)
    
    # Define paths
    model_path = "../models/skin_disease_model.h5"
    data_dir = "../data/skin_dataset" # Updated path for HAM10000
    # Updated class names for HAM10000
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        print("Please train the model first.")
        return
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        return
    
    # Create data generators
    print("Loading test data...")
    # Note: We use the validation set for evaluation in this example
    # You could also create a separate test generator if you have a dedicated test set
    _, test_gen = create_data_generators(data_dir, batch_size=32)
    
    # Evaluate model
    print("Evaluating model...")
    report, cm = evaluate_model(model_path, test_gen, class_names)
    
    # Save evaluation results
    os.makedirs('../documentation', exist_ok=True)
    
    # Save classification report
    with open('../documentation/classification_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n")
        for class_name in class_names:
            f.write(f"{class_name} - Precision: {report[class_name]['precision']:.4f}, "
                    f"Recall: {report[class_name]['recall']:.4f}, "
                    f"F1-Score: {report[class_name]['f1-score']:.4f}\n")
        f.write(f"Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}\n")
    
    print("\nEvaluation completed. Results saved to documentation folder.")

if __name__ == "__main__":
    main()