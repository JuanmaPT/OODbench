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
class_selection=  [stingray, junco, bullfrog]
config = Configuration(model= "ViT", 
                       N = 1,
                       id_classes= class_selection,
                       resolution= 10,
                       dataset = "ImageNetVal_small",
                       )

filenamesCombis =  getCombiFromDBoptimal(config)
planesets = []

for i, pathImgs in tqdm(enumerate(filenamesCombis), total=len(filenamesCombis), desc="Processing"):
    planesets.append(Planeset(Triplet(pathImgs, config), config))

descriptors = [PlanesetInfoExtractor(planeset,config) for planeset in planesets]
for planeset in planesets:
    planeset.show()

"""margin_dict = {label: [] for label in config.labels}
for i,planeset in enumerate(planesets):
    for j in range(3):
        if planeset.triplet.true_label[j] == planeset.triplet.prediction[j]:
            margin_dict[planeset.triplet.true_label[j]].append(descriptors[i].margin[j])

for classLabel, values in margin_dict.items():
    print(classLabel)
    #if values:  # Check if the list is not empty
    plot_pmf(values, classLabel, 15, config, 0, 20)"""

 
    
#%% Visualization Example

