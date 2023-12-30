# TRAINING MY OWN DECODER 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from PIL import Image

import torch.optim as optim
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform = None):
        self.data_folder = data_folder
        self.transform = transform
        self.resnet18_model = resnet18(pretrained=True)
        self.resnet18_model = torch.nn.Sequential(*(list(self.resnet18_model.children())[:-1]))
        self.resnet18_model.eval()

        self.image_dataset = ImageFolder(root=data_folder)

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        img_path, _ = self.image_dataset.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        # Extract features using ResNet18
        with torch.no_grad():
            latent_vector = self.resnet18_model(img.unsqueeze(0))

  
        print(latent_vector.size())
        #latent_vector = latent_vector.view(latent_vector.size(0), -1, 1, 1)
        return {'latent_vector': latent_vector.squeeze(0), 'image': img}


class ResNetDecoder(nn.Module):

    def __init__(self):
        super(ResNetDecoder, self).__init__()

        self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=2)
        self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=2)
        self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64, layers=2)
        self.conv4 = DecoderResidualBlock(hidden_channels=64, output_channels=64, layers=2)

        # Adjust the output channels of conv5 to match the expected channels by the subsequent layers
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)
        return x


class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels, output_channels, layers):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):
            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels, upsample=True)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels, upsample=False)

            self.add_module('%02d DecoderResidualLayer' % i, layer)

    def forward(self, x):
        for _, layer in self.named_children():
            x = layer(x)
        return x


class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels, output_channels, upsample):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)                
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2, output_padding=1, bias=False)   
            )
        else:
            self.upsample = None
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x


#%%
data_folder = "C:/Users/Blanca/Documents/GitHub/OODbench/smallDatasets/ImageNetVal_small/"
transform=transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to the same size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomDataset(data_folder, transform)

#split datset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

#%% Model and training parameters
decoder = ResNetDecoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

#%% Model Training 
best_val_loss = float('inf')
# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    decoder.train()  # Set model to training mode
    for batch in train_dataloader:
        latent_vectors = batch['latent_vector']
        target_images = batch['image']

        # Forward pass
        output_images = decoder(latent_vectors)

        # Compute the loss
        loss = criterion(output_images, target_images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Validation loop
    decoder.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_loss = 0.0
        for batch in val_dataloader:
            latent_vectors = batch['latent_vector']
            target_images = batch['image']

            # Forward pass
            output_images = decoder(latent_vectors)

            # Compute the loss
            val_loss += criterion(output_images, target_images).item()

        average_val_loss = val_loss / len(val_dataloader)
        print(f'Validation - Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_val_loss:.4f}')
        # Check if the current model has the best validation loss
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss

            # Save the model weights when the validation loss is the best
            torch.save(decoder.state_dict(), 'best_resnet_decoder_weights.pth')
            print('Saved the model with the best validation loss.')

        
#%% Inference example

latent_example = dataset[300]['latent_vector']
decoder.eval()
with torch.no_grad():
    generated_image = decoder(latent_example)  

generated_image = generated_image.squeeze(0)  # Remove batch dimension
generated_image = generated_image.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
generated_image = generated_image.numpy()  # Convert PyTorch tensor to NumPy array
generated_image = np.clip(generated_image, 0, 1)

# Convert NumPy array to PIL image
pil_image = Image.fromarray((generated_image * 255).astype(np.uint8))

# Example: Show the PIL image
pil_image.show()  

 
        

