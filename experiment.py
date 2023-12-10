from utils import *
from TripletExtractor import Triplet
from PlanesetExtractor import Planeset
from InfoDBExtractor import PlanesetInfoExtractor
from tqdm import tqdm
import numpy as np

# Dataset, model, class selection 
# Meter esto en un arg parser al final

datasets = ["ImageNetA_small", "ImageNetSD_small", "ImageNetVal_small",  "SignalBlur_small"]
models = ["ResNet18", "ViT"]


stingray = "n01498041"
junco = "n01534433"
robin= "n01558993"
jay= "n01580077"
bad_eagle= "n01614925"
bullfrog = "n01641577"
agama = "n01687978"

#datasets = ["ImageNetSD_small"]
#models = ["ResNet18"]
class_selection=  [stingray, junco, bullfrog, agama, robin, jay, bad_eagle]

#result_folder_name = 'MarginRes'
#result_folder_name = create_result_folder(result_folder_name)
create_result_folder(models)
save_info = True
save_plot = True

if save_info:
    margin_ResNet18 = {}
    margin_ViT = {}
    
#marging_save_list = []
for dataset in datasets:
    for model in models:
        config = Configuration(model= model, 
                               N = 1, 
                               id_classes= class_selection,
                               resolution= 10,
                               dataset = dataset
                               )
    
        filenamesCombis =  getCombiFromDBoptimalGoogleDrive(config)
        #margins_save = [] #list for storing the marging values
        #margins_save.append(model)
        #margins_save.append(dataset)
        
        # planeset computation per triplet
        planesets = []
        for i, pathImgs in tqdm(enumerate(filenamesCombis), total=len(filenamesCombis), desc="Processing"):
            triplet_object = Triplet(pathImgs, config)
            planeset_object = Planeset(triplet_object, config)
            planesets.append(planeset_object)
        
        # margin extraction
        descriptors = [PlanesetInfoExtractor(planeset,config) for planeset in planesets]
        margin_dict = {label: [] for label in config.labels}
        for i,planeset in enumerate(planesets):
            for j in range(3):
                if planeset.triplet.true_label[j] == planeset.triplet.prediction[j]:
                    margin_dict[planeset.triplet.true_label[j]].append(round(descriptors[i].margin[j],4))
        
        if save_info:
            if model == "ResNet18":
                margin_ResNet18[dataset] = margin_dict
            if model == "ViT":
                margin_ViT[dataset] = margin_dict
      
        if save_plot:
            for classLabel, values in margin_dict.items():
                plot_pmf(values, classLabel, 15, config, 0, 20, "result/plots")
        
        #marging_save_list.append(margins_save)


if save_info:
    if model == "ResNet18":
        save_to_csv(margin_ResNet18, "results/ResNet18/values/")
    if model == "ViT":
        save_to_csv(margin_ViT, "results/ViT/values/")

print('All done')
#%%













