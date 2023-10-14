import os
import random
import shutil

# Set the source and destination directories
source_dir = '/home/juanma/TRDP/OODatasets/Signal/motion_blur/1'
destination_dir = '/home/juanma/TRDP/OODrepo/dbViz/OODatasets/signal'

# Set the classes you want to select images from
classes = ['n02106662', 'n03388043', 'n03594945']

# Set the number of images you want to select from each class
num_images_per_class = 5

# Iterate over the classes
for class_name in classes:
    # Get the list of image files in the class directory
    class_dir = os.path.join(source_dir, class_name)
    image_files = os.listdir(class_dir)
    
    # Select random images from the list
    selected_images = random.sample(image_files, num_images_per_class)
    
    # Create the destination directory if it does not exist
    destination_class_dir = os.path.join(destination_dir, class_name)
    os.makedirs(destination_class_dir, exist_ok=True)
    
    # Move the selected images to the destination directory
    for image_name in selected_images:
        # Handle different file extensions
        file_name, file_extension = os.path.splitext(image_name)
        if file_extension.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        source_path = os.path.join(class_dir, image_name)
        destination_path = os.path.join(destination_class_dir, image_name)
        shutil.copy(source_path, destination_path)
