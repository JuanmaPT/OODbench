# -*- coding: utf-8 -*-
from utils import * 
from SomepalliFunctions import get_plane, plane_dataset
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.lines import Line2D

class Planeset:
    def __init__(self, triplet, config):
        self.triplet = triplet
        self.config = config
        self.planeset = self.computePlaneset()
        self.prediction, self.score = self.predict()
        self.anchors = self.getAnchors()
        self.predictedClasses = np.unique(self.prediction)
    
       
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
        
        planeset_pred = np.array(preds).reshape(r,r)
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
    
        
        
    def show(self,cmap,color_dict, title=None):

        # Asigning the new predicted classes
        unique_classes = np.unique(self.prediction)   
        num_classes = len(unique_classes)

        # Create a color map with black background
        color_map = np.zeros((self.prediction.shape[0], self.prediction.shape[1], 3))
        color_map_flat = np.zeros((self.prediction.shape[0], self.prediction.shape[1], 3))

        # Generate colors for each class
        #cmap = plt.get_cmap('rainbow')
        #colors = [to_rgba(cmap(i))[:3] for i in np.linspace(0, 1, num_classes)]
        custom_colors = get_custom_colors(num_classes)

        """
        # Assign colors based on prediction scores
        for class_label, color in zip(unique_classes, custom_colors):
            class_indices = np.where(self.prediction == class_label)
            class_scores = self.score[class_indices]
            
            #normalized_scores = (class_scores - np.min(class_scores)) / (np.max(class_scores) - np.min(class_scores))
            
            for idx, score in zip(zip(*class_indices), class_scores):
                color_map[idx] = np.array(color) * score # Adjust color intensity based on the prediction score
        """
         # Assign colors based on prediction scores
        for class_label in unique_classes:
            class_indices = np.where(self.prediction == class_label)
            class_scores = self.score[class_indices]

            for idx, score in zip(zip(*class_indices), class_scores):
                color = color_dict[class_label][0:3]  # Get color from the provided dictionary
                color_map[idx] = np.array(color) * score   # Adjust color intensity based on the prediction score
                color_map_flat[idx] = np.array(color)


        ##########################################################################33
        with open( "imagenet_class_index.json", 'r') as json_file:
            data = json.load(json_file)

        keys = list(self.anchors.keys())
        if self.config.resolution == 10:
            size= 0.1
        if self.config.resolution == 50:
            size= 1

        square1 = patches.Rectangle(self.anchors[keys[0]], width=size, height=size, fill=True, color='black') 
        square2 = patches.Rectangle(self.anchors[keys[1]], width=size, height=size, fill=True, color='black')
        square3 = patches.Rectangle(self.anchors[keys[2]], width=size, height=size, fill=True, color='black')
        

        fig = plt.figure()
        #fig, (ax1,ax1_1, ax2, ax3_1, ax3_2, ax3_3) = plt.subplots(2, 3, figsize=(20, 5))
        gs = fig.add_gridspec(2, 3, hspace=0, wspace=0)
        (ax1, ax1_1, ax2), (ax3_1, ax3_2, ax3_3) = gs.subplots(sharex='col', sharey='row')


        # Display the color map
        im = ax1.imshow(color_map)
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title('Planeset prediction')
        ax1.axis('off')

        #Display colormap unaltered
        
    
        ax1.add_patch(square1)
        ax1.add_patch(square2)
        ax1.add_patch(square3)

        ax1_1.imshow(color_map_flat)

        
        # Create a horizontal color bar for each class (going from dark to bright)
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

        # Add text annotations for 0 and 1 below the bars
        #ax2.text(0, 0.25, '0', ha='center', va='center', fontsize=7)
        #ax2.text(0.5, 0.25, '1', ha='center', va='center', fontsize= 7) 
        ax2.text(0.45, 0.85, 'Prediction Scores colorbar', ha='center', va='center', fontsize= 9)
      
        
        # visualize triplet of images:
        for i, path in enumerate(self.triplet.pathImgs):
            img = mpimg.imread(path)
            ax3 = plt.subplot(1, 5, i + 3)
            ax3.imshow(img)
            ax3.axis('off')
            ax3.set_title(f"True class: {self.config.labels[i]}, {data[str(self.config.labels[i])][1]}\nPrediction:  {self.triplet.prediction[i]}, {data[str(self.triplet.prediction[i])][1]} \nScore:  {self.triplet.score[i]}")

        plt.show()


    def show_simple(self, cmap, color_dict, title=None):
        unique_classes = np.unique(self.prediction)
        num_classes = len(unique_classes)

        color_matrix = np.zeros((self.prediction.shape[0], self.prediction.shape[1], 3))
        color_matrix_flat = np.zeros((self.prediction.shape[0], self.prediction.shape[1], 3))

        for class_label in unique_classes:
            class_indices = np.where(self.prediction == class_label)
            class_scores = self.score[class_indices]
            for idx, score in zip(zip(*class_indices), class_scores):
                color = np.array(color_dict[class_label][0:3]) * score
                color_matrix[idx] = color
                color_matrix_flat[idx] = np.array(color_dict[class_label][0:3])

        keys = list(self.anchors.keys())
        size = 0.1 if self.config.resolution == 10 else 1

        fig, axes = plt.subplots(2, 3, figsize=(10, 6))

        axes[0, 0].imshow(color_matrix)
        axes[0, 0].set_title(title if title else 'Planeset prediction')
        axes[0, 0].axis('off')

        for key in keys:
            square = patches.Rectangle(self.anchors[key], width=size, height=size, fill=True, color='black')
            axes[0, 0].add_patch(square)

        axes[0, 1].imshow(color_matrix_flat, cmap='gray')
        axes[0, 1].set_title('Flat Prediction')
        axes[0, 1].axis('off')

        with open('imagenet1klabels.txt', 'r') as file:
            class_labels_dict = eval(file.read())

        legend_elements = [Line2D([0], [0], marker='o', color='w', label=class_labels_dict[class_label][0:13],
                                markerfacecolor=np.array(color_dict[class_label][0:3]), markersize=10)
                        for class_label in unique_classes]

        axes[0, 1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, borderaxespad=0.5)
        axes[0, 2].axis('off')

        with open( "imagenet_class_index.json", 'r') as json_file:
            data = json.load(json_file)

        # Visualize triplet of images in the second row
        for i, path in enumerate(self.triplet.pathImgs):
            img = mpimg.imread(path)
            
            # Plot the image
            axes[1, i].imshow(img)
            axes[1, i].axis('off')
            
            # Add circle in the top-left corner
            true_class = self.config.labels[i]
            try:
                color = np.array(color_dict[true_class][0:3])
                circle = patches.Circle((0.05, 0.95), radius=0.03, transform=axes[1, i].transAxes, edgecolor=color, facecolor='none', lw=2)
                axes[1, i].add_patch(circle)
            except:
                print("Missing value")
            
            
            # Set title
            axes[1, i].set_title(f"True class: {self.config.labels[i]}, {data[str(self.config.labels[i])][1]}\n...
                                 Prediction: {self.triplet.prediction[i]}, {data[str(self.triplet.prediction[i])][1]}\n...
                                 Score: {self.triplet.score[i]}", fontsize='small', y=-0.35)

        # Add arrows from the anchor subplots to the anchor squares
        for i, key in enumerate(keys):
            x_center, y_center = self.anchors[key][0] + size / 2, self.anchors[key][1] + size / 2
            axes[0, 2].annotate('', xy=(x_center, y_center), xytext=(0.5, 0.5),
                                arrowprops=dict(facecolor='black', shrink=0.05))

        plt.show()
        print('Representation completed!')