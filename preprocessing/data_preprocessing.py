"""
Data Preprocessing Module for Skin Disease Detection System (HAM10000 - 7 Classes)
CIT 4299: IT Project
"""
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for model input
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for the image (width, height)
    
    Returns:
        numpy.ndarray: Preprocessed image array or None if error
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def create_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    """
    Create train and validation data generators with augmentation for HAM10000 (7 classes).
    
    Args:
        data_dir (str): Path to the directory containing class folders
        img_size (tuple): Image size for resizing
        batch_size (int): Batch size for training
    
    Returns:
        tuple: (train_generator, validation_generator)
    """
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for validation set
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical', # For multi-class
        subset='training',
        seed=42,
        # classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] # Optional: explicitly define class order if needed
    )
    
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical', # For multi-class
        subset='validation',
        seed=42,
        # classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] # Optional: explicitly define class order if needed
    )
    
    return train_generator, validation_generator

# Test the functions
if __name__ == "__main__":
    print("Data Preprocessing Module Ready")
    
    # Test with your organized HAM10000 dataset
    data_directory = "../data/skin_dataset" # Updated path for HAM10000
    
    if os.path.exists(data_directory):
        train_gen, val_gen = create_data_generators(data_directory)
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Number of classes: {train_gen.num_classes}")
        print(f"Class indices: {train_gen.class_indices}")
    else:
        print(f"Directory {data_directory} not found.")
        print("Please organize your HAM10000 dataset into class folders first.")