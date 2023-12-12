# -*- coding: utf-8 -*-
from utils import * 
from SomepalliFunctions import get_plane, plane_dataset
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import plotly.graph_objects as go


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
    
    def show(self,class_to_color,save_title):
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
            #color_map_flat[class_indices] = np.array(custom_colors[class_label])[:3]
            #color_map_scores[class_indices] = np.array(custom_colors[class_label])[:3]*class_scores[:, np.newaxis]
            print(class_to_color.get(class_label, None)[:3])
            color_map_flat[class_indices] = class_to_color.get(class_label, None)[:3]
            color_map_scores[class_indices] = class_to_color.get(class_label, None)[:3]*class_scores[:, np.newaxis]

    
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
        
        color_legends = [ np.array(class_to_color.get(class_label, None)[:3]) for class_label in unique_classes]
        #color_legends = [ np.array(custom_colors[class_label])[:3] for class_label in unique_classes]
        legend_patches = [mpatches.Patch(color=color, label=f"{class_label}: {data[str(class_label)][1]}") for class_label, color in zip(unique_classes, color_legends)]

        axes[0, 2].legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0, 1))
        axes[0, 2].axis('off')
       
        # visualize triplet of images:
        for i, path in enumerate(self.triplet.pathImgs):
            img = mpimg.imread(path)
            axes[1, i].imshow(img)
            axes[1, i].axis('off')
            axes[1, i].set_title(f"True class: {self.triplet.true_label[i]}, {data[str(self.triplet.true_label[i])][1]}\nPrediction:  {self.triplet.prediction[i]}, {data[str(self.triplet.prediction[i])][1]} \nScore:  {self.triplet.score[i]}")
    
       
        title_sup = f"{self.config.dataset}_{self.config.modelType}"
        fig.suptitle(title_sup, y=1, fontsize=20)
        fig.suptitle(title_sup, y=0.98, fontsize=20)
        fig.tight_layout()
        plt.show()
        #fig.savefig(save_title)
        #fig.clear()
        #plt.close()

    def show_scores(self,class_to_color,save_title):
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
            #color_map_flat[class_indices] = np.array(custom_colors[class_label])[:3]
            #color_map_scores[class_indices] = np.array(custom_colors[class_label])[:3]*class_scores[:, np.newaxis]
            print(class_to_color.get(class_label, None)[:3])
            color_map_flat[class_indices] = class_to_color.get(class_label, None)[:3]
            color_map_scores[class_indices] = class_to_color.get(class_label, None)[:3]*class_scores[:, np.newaxis]

    
        fig,axes = plt.subplots(1,1, figsize=(15, 10))
    
        # Display planeset predictions + scores
        axes.imshow(color_map_scores)
        axes.set_title('Planeset Prediciton Scores')
        axes.axis('off')
        
        
            
        # add a legend 
        with open("imagenet_class_index.json", 'r') as json_file:
            data = json.load(json_file)
    
        
        # Add a legend to axes[0, 2] with labels showing the values corresponding to class numbers
        legend_labels = [f"{class_label}: {data[str(class_label)][1]}" for class_label in unique_classes]
        
        color_legends = [ np.array(class_to_color.get(class_label, None)[:3]) for class_label in unique_classes]
        #color_legends = [ np.array(custom_colors[class_label])[:3] for class_label in unique_classes]
        legend_patches = [mpatches.Patch(color=color, label=f"{class_label}: {data[str(class_label)][1]}") for class_label, color in zip(unique_classes, color_legends)]

        axes.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0, 1))
        axes.axis('off')
       
    
       
        title_sup = f"{self.config.dataset}_{self.config.modelType}"
        fig.suptitle(title_sup, y=1, fontsize=20)
        fig.suptitle(title_sup, y=0.98, fontsize=20)
        fig.tight_layout()
        plt.show()
        print('Showing')
        #fig.savefig(save_title)
        #fig.clear()
        #plt.close()


   


    def show_scores_3D(self, class_to_color):

        keys = list(self.anchors.keys())
        anchor_index = []
        for key in keys:
            anchor_index.append(self.anchors[key])
        
        unique_classes = np.unique(self.prediction)
        scores_surfaces = np.zeros((3, len(unique_classes)))


        # Create an empty figure
        fig = go.Figure()

        for c, class_label in enumerate(unique_classes):
            class_indices = np.where(self.prediction == class_label)
            # Create a boolean matrix where True corresponds to the class_label
            class_mask = (self.prediction == class_label)
            # Use the boolean matrix to extract scores for the current class
            class_score_matrix = self.score.copy()  # You can use .copy() to avoid modifying the original array
            class_score_matrix[~class_mask] = 0  # Set scores
            color = class_to_color.get(class_label, [0, 0, 0])[:3]

            # Create a surface plot for the current class
            X, Y = np.meshgrid(np.arange(self.score.shape[0]), np.arange(self.score.shape[1]))
            surface = go.Surface(z=class_score_matrix, x=X, y=Y, colorscale=[[0, f'rgb({color[0]}, {color[1]}, {color[2]})'],
                                                                    [1, f'rgb({color[0]}, {color[1]}, {color[2]})']],
                                showscale=False)
            for i, key in enumerate(keys):
                scores_surfaces[i,c] = class_score_matrix[self.anchors[key]]
            
            # Add the surface to the figure
            fig.add_trace(surface)

        # Update trace and layout settings
        fig.update_traces(
            contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        )
        fig.update_layout(
            title='3D DB',
            width=1000,
            height=1000,
            margin=dict(l=65, r=50, b=65, t=90),
            scene=dict(
                xaxis_title='Projections',
                yaxis_title='Image size',
                zaxis_title='Intensities'
            )
        )

        #Plot points on anchor location

        
        for i in range(2):
            for key in keys:
                x_anchor, y_anchor = self.anchors[key]
                text_anchor = ['bottom', 'top'][i]  # Use 'bottom' or 'top' depending on the anchor iteration

                # Create a scatter plot for the anchor point with a circle marker
                fig.add_trace(go.Scatter3d(
                    x=[x_anchor],
                    y=[y_anchor],
                    z=[self.score[self.anchors[key]]],  # Adjust the z-coordinate as needed
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='black',  # You can change the color if needed
                        symbol='circle',
                    ),
                    text='Anchor',
                    textposition=f'{text_anchor} center',
                ))
                

        # Show the plot
        fig.show()
        print('Figure shown')

    def get_predictedClasses(self):
        return self.predictedClasses
    
        
        
