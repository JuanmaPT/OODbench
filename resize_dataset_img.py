import os
from PIL import Image

# Set the input folder path
folder_path = '/home/juanma/TRDP/OODrepo/dbViz/OODatasets/imagenet_val_original'

# Set the output folder path
output_folder = '/home/juanma/TRDP/OODrepo/dbViz/OODatasets/imagenet_val_resized'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all the images in the specified folder and its subfolders
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file is an image (you can add more image extensions if needed)
        if file.endswith('.jpg') or file.endswith('.JPEG') or file.endswith('.png'):
            # Get the file path
            file_path = os.path.join(root, file)
            
            # Open the image using PIL
            image = Image.open(file_path)
            
            # Calculate the aspect ratio
            width, height = image.size
            aspect_ratio = width / height
            
            # Resize the image while maintaining aspect ratio
            if aspect_ratio > 1:
                new_width = 500
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = 500
                new_width = int(new_height * aspect_ratio)
                
            resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
            
            # Create a new blank image with the desired size (500x500)
            padded_image = Image.new('RGB', (500, 500), (0, 0, 0))
            
            # Calculate the position to paste the resized image to maintain center alignment
            paste_x = int((500 - new_width) / 2)
            paste_y = int((500 - new_height) / 2)
            
            # Paste the resized image onto the blank image
            padded_image.paste(resized_image, (paste_x, paste_y))
            
            # Get the relative path from the input folder
            relative_path = os.path.relpath(file_path, folder_path)
            
            # Create the output directory structure and save the padded image
            output_path = os.path.join(output_folder, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            padded_image.save(output_path)