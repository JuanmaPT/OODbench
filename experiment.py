from utils import *
from TripletExtractor import Triplet
from PlanesetExtractor import Planeset
from InfoDBExtractor import PlanesetInfoExtractor
from tqdm import tqdm
import numpy as np

# Dataset, model, class selection 
# Meter esto en un arg parser al final

#datasets = ["ImageNetA_small", "ImageNetSD_small", "ImageNetVal_small",  "SignalBlur_small"]
#models = ["ResNet18", "ViT"]
#c1, c2, c3  = "n01498041","n01534433","n01687978"

stingray = "n01498041"
junco = "n01534433"
robin= "n01558993"
jay= "n01580077"
bad_eagle= "n01614925"
bullfrog = "n01641577"
agama = "n01687978"


datasets = ["ImageNetVal_small"]
models = ["ResNet18"]
class_selection=  [stingray, junco, bullfrog, agama, robin, jay, bad_eagle]


result_folder_name = 'CambioMargin'
result_folder_name = create_result_folder(result_folder_name)

save_info = False

#if save_info:
    #margin_ResNet18 = []
    #margin_ViT = []
    
#marging_save_list = []

for dataset in datasets:
    for model in models:
        config = Configuration(model= model, 
                               N = 1, 
                               id_classes= class_selection,
                               resolution= 10,
                               dataset = dataset
                               )
    
        filenamesCombis =  getCombiFromDBoptimal(config)
        margins_save = [] #list for storing the marging values
        margins_save.append(model)
        margins_save.append(dataset)
        planesets = []
        for i, pathImgs in tqdm(enumerate(filenamesCombis), total=len(filenamesCombis), desc="Processing"):
            triplet_object = Triplet(pathImgs, config)
            planeset_object = Planeset(triplet_object, config)
            planesets.append(planeset_object)
           
        descriptors = [PlanesetInfoExtractor(planeset,config) for planeset in planesets]
        
        #marginClass = [[] for _ in range(len(config.id_classes))]
        #for i in range(len(planesets)):
            #for j in range(len(config.id_classes)):
                #if planesets[i].triplet.isImgPredCorrect[j] == True:
                    #marginClass[j].append(descriptors[i].margin[j])
                    #margins_save.append(round(descriptors[i].margin[j], 4))
       
        margin_dict = {label: [] for label in config.labels}
        for i,planeset in enumerate(planesets):
            for j in range(3):
                if planeset.triplet.true_label[j] == planeset.triplet.prediction[j]:
                    margin_dict[planeset.triplet.true_label[j]].append(descriptors[i].margin[j])
        
        #if save_info:
            #if model == "ResNet18":
                #margin_ResNet18.append(marginClass)

            #if model == "ViT":
                #.append(marginClass)
      
        for classLabel, values in margin_dict.items():
            print(classLabel)
            #if values:  # Check if the list is not empty
            plot_pmf(values, classLabel, 15, config, 0, 20)
        
        #marging_save_list.append(margins_save)
    
#save_to_csv(marging_save_list, f"results\{result_folder_name}\margin_ResNet18.csv")
print('All done')
#%%













