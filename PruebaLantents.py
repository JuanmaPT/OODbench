from utils import *
from TripletExtractor import Triplet
from PlanesetExtractor import Planeset
from InfoDBExtractor import PlanesetInfoExtractor
import numpy as np

def load_latent_example(idx, planeset):
    input_tensor = planeset_obj.planeset[idx]
    upsample_layer = nn.Upsample(size=(7, 7), mode='nearest')
    upsample_layer(input_tensor)
    return upsampleTensor(latent_example)
    

class2id = {
        "stingray": ["n01498041", "6"] ,
        "junco": ["n01534433", "13"],
        "robin": ["n01558993", "15"],
        "jay": ["n01580077", "17"],
        "bad_eagle": ["n01614925", "22"] ,
        "bullfrog": ["n01641577", "30"],
        "agama": ["n01687978", "42"]
    }
    
class_selection = []
for class_ in ["stingray", "junco", "bullfrog"]:
    class_selection.append(class2id[class_][0])
        
config = Configuration(model= "ResNet18", 
                        N = 1, 
                        dataset = "ImageNetVal_small" ,
                        id_classes= class_selection,
                        resolution= 20,
                        useFilteredPaths = "False", 
                        )

filenamesCombis =  getCombiFromDBoptimalGoogleDrive(config)
triplet_obj= Triplet(filenamesCombis[0], config)
planeset_obj = Planeset(triplet_obj, config)
latent_example = load_latent_example(50, planeset_obj)







