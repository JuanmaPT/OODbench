import torchvision
import torchvision.transforms as transforms
import torch
import random
import os
from PIL import Image

def get_plane(img1, img2, img3):
    ''' Calculate the plane (basis vecs) spanned by 3 images
    Input: 3 image tensors of the same size
    Output: two (orthogonal) basis vectors for the plane spanned by them, and
    the second vector (before being made orthogonal)
    '''

    # Step 1: Verify tensor sizes
    if img1.size() != img2.size() or img1.size() != img3.size():
        min_size = min(img1.size(), img2.size(), img3.size())
        min_height, min_width = min_size[1], min_size[2]  # Extract minimum height and width
        aspect_ratio = min_height / min_width  # Calculate aspect ratio
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((min_height, min_width)),  # Resize to the minimum common size
            transforms.Pad((0, 0, 0, int((min_width - min_height * aspect_ratio) / 2)), fill=0)  # Pad to achieve 1:1 aspect ratio
        ])
        img1 = transform(img1)
        img2 = transform(img2)
        img3 = transform(img3)
 


    # Step 2: Subtract img1 from img2 and img3
    a = img2 - img1
    b = img3 - img1

    # Step 3: Check and resize a and b if necessary
    if a.size() != b.size():
        min_size = min(a.size(), b.size())
        min_height, min_width = min_size[1], min_size[2]  # Extract minimum height and width
        aspect_ratio = min_height / min_width  # Calculate aspect ratio
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((min_height, min_width)),  # Resize to the minimum common size
            transforms.Pad((0, 0, 0, int((min_width - min_height * aspect_ratio) / 2)), fill=0)  # Pad to achieve 1:1 aspect ratio
        ])
        a = transform(a)
        b = transform(b)

    # Step 4: Use a and b in the ResNet model for image classification
    # ... (code for using a and b in the ResNet model)
    #####################################################33
    a_norm = torch.dot(a.flatten(), a.flatten()).sqrt()
    a = a / a_norm
    first_coef = torch.dot(a.flatten(), b.flatten())
    #first_coef = torch.dot(a.flatten(), b.flatten()) / torch.dot(a.flatten(), a.flatten())
    b_orthog = b - first_coef * a
    b_orthog_norm = torch.dot(b_orthog.flatten(), b_orthog.flatten()).sqrt()
    b_orthog = b_orthog / b_orthog_norm
    second_coef = torch.dot(b.flatten(), b_orthog.flatten())
    #second_coef = torch.dot(b_orthog.flatten(), b.flatten()) / torch.dot(b_orthog.flatten(), b_orthog.flatten())
    coords = [[0,0], [a_norm,0], [first_coef, second_coef]]

    return a, b_orthog, b, coords


class plane_dataset(torch.utils.data.Dataset):
    def __init__(self, base_img, vec1, vec2, coords, resolution,
                    range_l=.1, range_r=.1):
 
        self.base_img = base_img
        self.vec1 = vec1
        self.vec2 = vec2
        self.coords = coords
        self.resolution = resolution
        x_bounds = [coord[0] for coord in coords]
        y_bounds = [coord[1] for coord in coords]

        self.bound1 = [torch.min(torch.tensor(x_bounds)), torch.max(torch.tensor(x_bounds))]
        self.bound2 = [torch.min(torch.tensor(y_bounds)), torch.max(torch.tensor(y_bounds))]
        
        self.grid = None

        len1 = self.bound1[-1] - self.bound1[0]
        len2 = self.bound2[-1] - self.bound2[0]
        
        #list1 and list2, which represent the grid of points in the 2D plane.

        list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, int(resolution))
        list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, int(resolution))

        self.grid = torch.meshgrid([list1,list2])

        self.coefs1 = self.grid[0].flatten()
        self.coefs2 = self.grid[1].flatten()
        
        
    def get_grid(self):
        return self.grid
                   
    def __len__(self):
        
        return self.coefs1.shape[0]

    def __getitem__(self, idx):
         generated_image = self.base_img + self.coefs1[idx] * self.vec1 + self.coefs2[idx] * self.vec2
         return generated_image

