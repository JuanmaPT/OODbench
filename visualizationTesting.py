#%% 
from utils import *
from TripletExtractor import Triplet
from PlanesetExtractor import Planeset
from InfoDBExtractor import PlanesetInfoExtractor
from tqdm import tqdm
import numpy as np


#selection of the class id
stingray = "n01498041"
junco = "n01534433"
robin= "n01558993"
jay= "n01580077"
bad_eagle= "n01614925"
bullfrog = "n01641577"
agama = "n01687978"

class_selection=  [stingray, junco, bullfrog, agama, robin, jay, bad_eagle]
config = Configuration(model= "ResNet18", 
                       N = 1,
                       id_classes= class_selection,
                       resolution= 200,
                       dataset = "ImageNetVal_small",
                       )

filenamesCombis =  getCombiFromDBoptimalGoogleDrive(config)
planesets = []

for i, pathImgs in tqdm(enumerate(filenamesCombis), total=len(filenamesCombis), desc="Processing"):
    if i == 4:
        planesets.append(Planeset(Triplet(pathImgs, config), config))



descriptors = [PlanesetInfoExtractor(planeset,config) for planeset in planesets]
class_to_color = get_diff_color(planesets)

create_folder(f"results/{config.modelType}_res{config.resolution}")

#%% 
for i, planeset in enumerate(planesets):
    i = 4
    save_planeset_path = f"results/{config.modelType}_res{config.resolution}/planeset_{i}.png"
    print(save_planeset_path)
    #planeset.show_scores(class_to_color,save_planeset_path)
    planeset.show_scores_3D(class_to_color)
    planeset.show(class_to_color,'hello')
    

"""margin_dict = {label: [] for label in config.labels}
for i,planeset in enumerate(planesets):
    for j in range(3):
        if planeset.triplet.true_label[j] == planeset.triplet.prediction[j]:
            margin_dict[planeset.triplet.true_label[j]].append(descriptors[i].margin[j])

for classLabel, values in margin_dict.items():
    print(classLabel)
    #if values:  # Check if the list is not empty
    plot_pmf(values, classLabel, 15, config, 0, 20)"""



# %%
import plotly.graph_objects as go
def show_scores_3D(planeset, class_to_color):

    keys = list(planeset.anchors.keys())
    unique_classes = np.unique(planeset.prediction)

    # Create an empty figure
    fig = go.Figure()

    with open("imagenet_class_index.json", 'r') as json_file:
        class_index = json.load(json_file)

    for c, class_label in enumerate(unique_classes):
        class_indices = np.where(planeset.prediction == class_label)
        # Create a boolean matrix where True corresponds to the class_label
        class_mask = (planeset.prediction == class_label)
        # Use the boolean matrix to extract scores for the current class
        class_score_matrix = planeset.score.copy()  # You can use .copy() to avoid modifying the original array
        class_score_matrix[~class_mask] = 0  # Set scores
        color = class_to_color.get(class_label, [0, 0, 0])[:3]
        color = [round(value * 255) for value in color]

        # Create a surface plot for the current class
        X, Y = np.meshgrid(np.arange(planeset.score.shape[0]), np.arange(planeset.score.shape[1]-1, -1, -1))
        surface = go.Surface(z=class_score_matrix, x=X, y=Y,
                            colorscale=[[0, f'rgb(0,0,0)'],
                                        [1, f'rgb({color[0]}, {color[1]}, {color[2]})']],
                                        showscale=False,
                                        )
        print(color)
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
            xaxis_title='xaxis decision space',
            yaxis_title='yaxis decision space',
            zaxis_title='Score'
        )
    )

    for c, class_label in enumerate(unique_classes):
        fig.add_trace(go.Scatter3d(
                x=[(0.3)],
                y=[(0.3)],
                z = [(0.3)],
                mode='markers',
                marker=dict(size=12, color=f'rgb({color[0]}, {color[1]}, {color[2]})'),
                name= f"{class_label}: {class_index[str(class_label)][1]}"
            ))


    for i,key in enumerate(keys):
        x_anchor, y_anchor = planeset.anchors[key]
        text_anchor = ['bottom', 'top','middle'][i]  # Use 'bottom' or 'top' depending on the anchor iteration

        # Create a scatter plot for the anchor point with a circle marker
        fig.add_trace(go.Scatter3d(
            x=[x_anchor],
            y=[planeset.score.shape[1] - 1 - y_anchor],
            z=[planeset.score[y_anchor,x_anchor]+0.1],  # Adjust the z-coordinate as needed
            mode='markers',
            marker=dict(
                size=10,
                color=['yellow', 'black', 'pink'][i],  # You can change the color if needed
                symbol=['circle', 'circle', 'diamond'][i],
            ),
            text=f'Anchor_{i}',
            textposition=f'{text_anchor} center',
        ))
        # Adding a line trace to create a dotted line in the z-direction
        fig.add_trace(go.Scatter3d(
            x=[x_anchor,x_anchor],
            y=[planeset.score.shape[1] - 1 - y_anchor,planeset.score.shape[1] - 1 - y_anchor],
            z=[planeset.score[y_anchor, x_anchor] + 0.1, 0], # Use the linspace values directly
            mode='lines',
            line=dict(color=['yellow', 'black', 'pink'][i]
                      , dash='solid',
                        width=6),  # You can customize the color and dash pattern
        ))
            
    # Show the plot
    fig.show()
    print('Figure shown')

show_scores_3D(planesets[0], class_to_color)
# %%