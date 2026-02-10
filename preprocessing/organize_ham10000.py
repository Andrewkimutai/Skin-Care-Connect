"""
Script to organize HAM10000 dataset by class using metadata.csv
CIT 4299: IT Project
"""
import pandas as pd
import os
import shutil
from pathlib import Path

def organize_ham10000_dataset():
    """
    Organize HAM10000 dataset by class based on metadata.csv
    """
    # Define paths (adjust if your data is in a different location)
    base_path = "../data"  # Path relative to this script's location (preprocessing/)
    images_part1_path = os.path.join(base_path, "HAM10000_images_part_1")
    images_part2_path = os.path.join(base_path, "HAM10000_images_part_2")
    metadata_path = os.path.join(base_path, "HAM10000_metadata.csv")
    output_path = os.path.join(base_path, "skin_dataset")
    
    # Check if required files exist
    if not os.path.exists(images_part1_path):
        print(f"Error: Images Part 1 folder not found at {images_part1_path}")
        return False
    
    if not os.path.exists(images_part2_path):
        print(f"Error: Images Part 2 folder not found at {images_part2_path}")
        return False
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return False
    
    # Read metadata
    try:
        df = pd.read_csv(metadata_path)
        print(f"Loaded metadata with {len(df)} entries")
        print(f"Available columns: {list(df.columns)}")
        
        # Check if 'dx' column exists (this is the diagnosis/class column)
        if 'dx' not in df.columns:
            print("Error: 'dx' column not found in metadata. This column should contain the class labels.")
            return False
            
        # Get unique classes
        classes = df['dx'].unique()
        print(f"Found {len(classes)} unique classes: {classes}")
        
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return False
    
    # Create output directory and class folders
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    for class_name in classes:
        class_folder = os.path.join(output_path, class_name)
        Path(class_folder).mkdir(exist_ok=True)
    
    # Process images from Part 1 and Part 2
    processed_count = 0
    error_count = 0
    
    print("\nProcessing images...")
    for _, row in df.iterrows():
        try:
            image_id = str(row['image_id'])
            diagnosis = str(row['dx'])
            
            # Try to find image in Part 1
            src_path = os.path.join(images_part1_path, image_id + '.jpg')
            
            if not os.path.exists(src_path):
                # If not found in Part 1, try Part 2
                src_path = os.path.join(images_part2_path, image_id + '.jpg')
                
                if not os.path.exists(src_path):
                    print(f"Warning: Image {image_id}.jpg not found in either part")
                    error_count += 1
                    continue
            
            # Destination path
            dst_path = os.path.join(output_path, diagnosis, image_id + '.jpg')
            
            # Copy image to class folder
            shutil.copy2(src_path, dst_path)
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images...")
                
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            error_count += 1
    
    print(f"\nDataset organization complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors: {error_count} images")
    return True

if __name__ == "__main__":
    print("HAM10000 Dataset Organization Tool")
    print("=" * 40)
    
    success = organize_ham10000_dataset()
    if success:
        print("\nDataset organized successfully!")
        print("You can now use the data with ImageDataGenerator for training.")
    else:
        print("\nDataset organization failed. Please check the error messages above.")