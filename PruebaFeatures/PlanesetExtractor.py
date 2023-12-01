# -*- coding: utf-8 -*-
from utils import * 
from SomepalliFunctions import get_plane, plane_dataset
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np

class Planeset:
    def __init__(self, triplet, config):
        self.triplet = triplet
        self.config = config
        self.planeset = self.computePlaneset()
        self.prediction, self.score = self.predict()
        self.anchors = self.getAnchors()
        self.predictedClasses = np.unique(self.prediction).astype(np.uint16)
    
       
    def computePlaneset(self):
        # calculate the plane spanned by the three images 
        a, b_orthog, b, coords = get_plane(self.triplet.features[0], self.triplet.features[1], self.triplet.features[2])
        # get the dataset of images on a 2D plane by the combination of the features
        return plane_dataset(self.triplet.features[0], a, b_orthog, coords, resolution= self.config.resolution )
        
    
    def predict(self):
        "returns 2D labelled image"
        self.config.model.eval()
        r = self.config.resolution
        preds = []
        scores = []
        for idx in range(len(self.planeset)):
            mixfeat = self.planeset[idx]
            #print("pred", mixfeat.shape)
            with torch.no_grad():
                pred = self.config.head_model(mixfeat)
                pred_class = pred.argmax().item()
                preds.append(pred_class)
                scores.append(pred.softmax(dim=-1).max().item())
        
        planeset_pred = np.array(preds).reshape(r,r).astype(np.uint16)
        planeset_score = np.array(scores).reshape(r,r)
                
        return planeset_pred, planeset_score
        
    def getAnchors(self):
        anchor_dict = {}
        
        distances = []  # List to store the distances
        triplet_index = []
        for image_idx in range(3):
            for batch_idx, inputs in enumerate(self.planeset):
                distance = torch.dist(inputs, self.triplet.features[image_idx])
                distances.append(distance.item()) 
            min_distance_index = distances.index(min(distances))           
            distances = [] 
            triplet_index.append(min_distance_index)
        
        for i,idx in enumerate(triplet_index):
            # convert 1d idx to x,y coords in a 2d grid
            x = idx % self.planeset.resolution
            y = idx // self.planeset.resolution
            
            # create the dictionary 
            anchor_dict[self.triplet.prediction[i]] = (x,y)
        
        return anchor_dict
    
        
        
    def show(self, title=None):
        unique_classes = np.unique(self.prediction)   
        num_classes = len(unique_classes)

        # Create a color map with black background
        color_map = np.zeros((self.prediction.shape[0], self.prediction.shape[1], 3))

        # Generate colors for each class
        #cmap = plt.get_cmap('rainbow')
        #colors = [to_rgba(cmap(i))[:3] for i in np.linspace(0, 1, num_classes)]
        custom_colors = get_custom_colors(num_classes)

        # Assign colors based on prediction scores
        for class_label, color in zip(unique_classes, custom_colors):
            class_indices = np.where(self.prediction == class_label)
            class_scores = self.score[class_indices]
            
            #normalized_scores = (class_scores - np.min(class_scores)) / (np.max(class_scores) - np.min(class_scores))
            
            for idx, score in zip(zip(*class_indices), class_scores):
                color_map[idx] = np.array(color) * score # Adjust color intensity based on the prediction score

        
        fig, (ax1, ax2, ax3_1, ax3_2, ax3_3) = plt.subplots(1, 5, figsize=(20, 5))

        # Display the color map
        im = ax1.imshow(color_map)
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title('Planeset prediction')
        ax1.axis('off')
        
       
        keys = list(self.anchors.keys())
        if self.config.resolution == 10:
            size= 0.1
        if self.config.resolution == 50:
            size= 1
        
        square1 = patches.Rectangle(self.anchors[keys[0]], width=size, height=size, fill=True, color='black') 
        square2 = patches.Rectangle(self.anchors[keys[1]], width=size, height=size, fill=True, color='black')
        square3 = patches.Rectangle(self.anchors[keys[2]], width=size, height=size, fill=True, color='black')
    
        ax1.add_patch(square1)
        ax1.add_patch(square2)
        ax1.add_patch(square3)

        with open( "imagenet_class_index.json", 'r') as json_file:
            data = json.load(json_file)
        
        
        # Create a horizontal color bar for each class (going from less bright to more bright)
        bar_height = 0.05  # Adjust the height based on your preference
        space_between_bars = 0.02  # Adjust the space between bars
        total_height = num_classes * (bar_height + space_between_bars) - space_between_bars
        start_y = (1 - total_height) / 2

        for i, (class_label, color) in enumerate(zip(unique_classes, custom_colors)):
            color_bar = np.ones((1, 100, 3)) * np.array(color)
            color_bar[0, :, :] *= np.linspace(0, 1, 100)[:, np.newaxis]  # Adjust color intensity (reversed)
            ax2.imshow(color_bar, extent=[0, 0.5, start_y + i * (bar_height + space_between_bars), start_y + (i + 1) * bar_height + i * space_between_bars], aspect='auto')

            # Label indicating the corresponding class, centered
            ax2.text(0.55, start_y + i * (bar_height + space_between_bars) + bar_height / 2, f'{class_label}: {data[str(class_label)][1]}', ha='left', va='center', rotation=0, fontsize=10)

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off') 
        ax2.text(0.45, 0.85, 'Prediction Scores colorbar', ha='center', va='center', fontsize= 9)
      
        
        # visualize triplet of images:
        for i, path in enumerate(self.triplet.pathImgs):
            img = mpimg.imread(path)
            ax3 = plt.subplot(1, 5, i + 3)
            ax3.imshow(img)
            ax3.axis('off')
            ax3.set_title(f"True class: {self.config.labels[i]}, {data[str(self.config.labels[i])][1]}\nPrediction:  {self.triplet.prediction[i]}, {data[str(self.triplet.prediction[i])][1]} \nScore:  {self.triplet.score[i]}")
        
        
        if title is None:
            title = f"{self.config.dataset} | {self.config.modelType}"
        fig.suptitle(title, y=1.05, fontsize=16)
        plt.show()