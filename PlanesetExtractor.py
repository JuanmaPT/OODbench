# -*- coding: utf-8 -*-
from utils import * 
from SomepalliFunctions import get_plane, plane_dataset
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

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
            mixfeat = mixfeat.squeeze()
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
    
    def show(self):
        unique_classes = np.unique(self.prediction)
        num_classes = len(unique_classes)
        
        # load the colors
        custom_colors = np.loadtxt('Colors1k.txt').tolist()
        # Assign colors based on predicted class and prediction scores
        color_map_scores = np.zeros((self.prediction.shape[0], self.prediction.shape[1], 3))
        color_map_flat = np.zeros((self.prediction.shape[0], self.prediction.shape[1], 3)) 
       
        for class_label in unique_classes:
            class_indices = np.where(self.prediction == class_label)
            class_scores = self.score[class_indices]
            color_map_flat[class_indices] = np.array(custom_colors[class_label])[:3]
            color_map_scores[class_indices] = np.array(custom_colors[class_label])[:3]*class_scores[:, np.newaxis] 

    
        fig,axes = plt.subplots(2,3, figsize=(15, 10))
    
        # Display planeset predictions + scores
        axes[0, 0].imshow(color_map_scores)
        axes[0, 0].set_title('Planeset Prediciton Scores')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(color_map_flat)
        axes[0, 1].set_title('Planeset Labels')
        axes[0, 1].axis('off')
        
        # and anchors
        keys = list(self.anchors.keys())
        size = 0.1 if self.config.resolution == 10 else 1
        for i in range(2):
            for key in keys:
                square = patches.Rectangle(self.anchors[key], width=size, height=size, fill=True, color='black')
                axes[0, i].add_patch(square)
            
            
        # add a legend 
        with open("imagenet_class_index.json", 'r') as json_file:
            data = json.load(json_file)
    
        
        # Add a legend to axes[0, 2] with labels showing the values corresponding to class numbers
        legend_labels = [f"{class_label}: {data[str(class_label)][1]}" for class_label in unique_classes]
        color_legends = [ np.array(custom_colors[class_label])[:3] for class_label in unique_classes]
        legend_patches = [mpatches.Patch(color=color, label=f"{class_label}: {data[str(class_label)][1]}") for class_label, color in zip(unique_classes, color_legends)]

        axes[0, 2].legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0, 1))
        axes[0, 2].axis('off')
       
        # visualize triplet of images:
        for i, path in enumerate(self.triplet.pathImgs):
            img = mpimg.imread(path)
            axes[1, i].imshow(img)
            axes[1, i].axis('off')
            axes[1, i].set_title(f"True class: {self.triplet.true_label[i]}, {data[str(self.triplet.true_label[i])][1]}\nPrediction:  {self.triplet.prediction[i]}, {data[str(self.triplet.prediction[i])][1]} \nScore:  {self.triplet.score[i]}")
    
       
        title_sup = f"{self.config.dataset} | {self.config.modelType}"
        fig.suptitle(title_sup, y=1.02, fontsize=20)
        fig.tight_layout()
        plt.show()
    
        
        
   