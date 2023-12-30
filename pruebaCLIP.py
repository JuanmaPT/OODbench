# -*- coding: utf-8 -*-
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def load_example():
    model = models.vgg16(pretrained=True)
    # Remove the last fully connected layer (classifier)
    base_model = nn.Sequential(*list(model.features.children())[:-1], nn.AdaptiveAvgPool2d(1))
    base_model.eval()
    
    path_example = "C:/Users/Blanca/Documents/GitHub/OODbench/smallDatasets/ImageNetVal_small/n01498041/ILSVRC2012_val_00001935_n01498041.jpeg"
    img = Image.open(path_example)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(img)
    
    # Expand dimensions to simulate batch size of 1
    input_batch = image.unsqueeze(0)
    with torch.no_grad():
        example = base_model(input_batch)
    
    print(example.size())
    return example

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

encoded_latent = load_example()

# Convert the encoded latent tensor to a string representation
encoded_latent_str = " ".join(map(str, encoded_latent.flatten().tolist()))

# Create a prompt for image generation
prompt = "Generate an image using the latent vector:"

# Combine prompt and encoded latent vector
input_text = f"{prompt} {encoded_latent_str}"

# Tokenize and process the input
inputs = processor(input_text, return_tensors="pt", padding=True, truncation=True)

# Specify pixel_values as a dummy tensor
dummy_pixel_values = torch.zeros((1, 3, 224, 224))  # replace with your desired size
inputs["pixel_values"] = dummy_pixel_values

# Generate image
with torch.no_grad():
    logits_per_image = model(**inputs).logits_per_image
    softmax_probs = logits_per_image.softmax(1)

# Convert softmax probabilities to a NumPy array
image_array = softmax_probs.cpu().numpy()

# Reshape to a 3D array (assuming you have a single image)
image_array = image_array.reshape(image_array.shape[1], image_array.shape[2])

# Rescale values from [0, 1] to [0, 255]
image_array = (image_array * 255).astype('uint8')

# Convert to PIL Image
image_pil = Image.fromarray(image_array)

# Display the image
image_pil.show()