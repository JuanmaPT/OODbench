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
                       resolution= 20,
                       dataset = "ImageNetVal_small",
                       )

filenamesCombis =  getCombiFromDBoptimalGoogleDrive(config)
planesets = []

for i, pathImgs in tqdm(enumerate(filenamesCombis), total=len(filenamesCombis), desc="Processing"):
    if i == 4:
        planesets.append(Planeset(Triplet(pathImgs, config), config))



descriptors = [PlanesetInfoExtractor(planeset,config) for planeset in planesets]

#%% 
class_to_color = get_diff_color(planesets)

create_folder(f"results/{config.modelType}_res{config.resolution}")

#%% 
for i, planeset in enumerate(planesets):
    i = 4
    save_planeset_path = f"results/{config.modelType}_res{config.resolution}/planeset_{i}.png"
    print(save_planeset_path)
    #planeset.show_scores(class_to_color,save_planeset_path)
    planeset.show_scores_3D(class_to_color)
    

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
