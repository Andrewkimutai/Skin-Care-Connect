"""
Model Training Module for Skin Disease Detection System (HAM10000 - 7 Classes)
CIT 4299: IT Project
"""
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the 'preprocessing' directory to the Python path for PyCharm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))

# Import the required function
from data_preprocessing import create_data_generators

class SkinDiseaseClassifier:
    def __init__(self, num_classes=7, img_size=(224, 224)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None
        
    def build_model(self, learning_rate=0.001):
        """
        Build transfer learning model using ResNet50 for HAM10000 (7 classes)
        """
        # Load pre-trained ResNet50 without top layers
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size[0], self.img_size[1], 3)
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x) # Output layer for 7 classes
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train_model(self, train_generator, validation_generator, epochs=25):
        """
        Train the model with callbacks
        """
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='../models/skin_disease_model_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def unfreeze_and_fine_tune(self, train_generator, validation_generator, 
                              fine_tune_epochs=10, fine_tune_lr=0.0001):
        """
        Unfreeze top layers and fine-tune with lower learning rate
        """
        # Get the base model layer (the ResNet50 part)
        base_model = self.model.layers[1] # This is the ResNet50 layer

        # Check if it's a functional or sequential model layer
        if hasattr(base_model, 'layers') and base_model.layers:
            # It's a functional model like ResNet50
            # Freeze all layers up to fine_tune_at
            fine_tune_at = 100
            print(f"Fine-tuning from layer {fine_tune_at} onwards...")
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            
            # Unfreeze the remaining layers
            for layer in base_model.layers[fine_tune_at:]:
                layer.trainable = True
        else:
            # It might be a Sequential model or not a composite layer as expected
            # This is less likely with ResNet50 but handle gracefully
            print("Base model structure unexpected for fine-tuning. Attempting to set trainable=True on base_model.")
            base_model.trainable = True # This will unfreeze the whole base model


        # Recompile with lower learning rate for fine-tuning
        # Use a much lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=fine_tune_lr), # Use the passed learning rate directly
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Starting fine-tuning...")
        # Continue training (fine-tuning)
        fine_tune_history = self.model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=validation_generator,
            verbose=1
        )
        
        return fine_tune_history
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('../documentation/training_history.png')
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

def main():
    """
    Main function to train the skin disease classifier using HAM10000 dataset
    """
    print("Starting Skin Disease Detection Model Training (HAM10000 - 7 Classes)")
    print("=" * 50)

    os.makedirs('../models', exist_ok=True)
    os.makedirs('../documentation', exist_ok=True) # Ensure documentation dir exists too

    # Create data generators
    data_dir = "../data/skin_dataset" # Updated path for HAM10000
    print(f"Loading data from: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        print("Please organize your HAM10000 dataset into class folders.")
        return

    train_gen, val_gen = create_data_generators(data_dir)
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Number of classes: {train_gen.num_classes}")
    print(f"Class indices: {train_gen.class_indices}")

    classifier = SkinDiseaseClassifier(num_classes=train_gen.num_classes)

    # Build model
    print("\nBuilding model...")
    model = classifier.build_model()
    print("Model built successfully!")
    print(f"Model summary:")
    model.summary()

    # Train model
    print("\nStarting model training...")
    history = classifier.train_model(train_gen, val_gen, epochs=20)

    # Fine-tune model
    print("\nStarting fine-tuning...")
    fine_tune_history = classifier.unfreeze_and_fine_tune(train_gen, val_gen, fine_tune_epochs=10)

    # Plot results
    print("\nPlotting training history...")
    classifier.plot_training_history()

    # Save model
    model_path = "../models/skin_disease_model.h5"
    print(f"\nSaving model to: {model_path}")
    classifier.save_model(model_path)

    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()