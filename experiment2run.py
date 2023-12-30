from utils import *
from TripletExtractor import Triplet
from PlanesetExtractor import Planeset
from InfoDBExtractor import PlanesetInfoExtractor
from tqdm import tqdm
import numpy as np
import argparse

#from memory_profiler import profile

# -Guardar las unique clases en cada triplet.
# -Añadir opción de filtrar por predicción 

# Availabe classes for the study
class2id = {
    "stingray":  ["n01498041", "6" ],
    "junco":     ["n01534433", "13"],
    "robin":     ["n01558993", "15"],
    "jay":       ["n01580077", "17"],
    "bad_eagle": ["n01614925", "22"],
    "bullfrog":  ["n01641577", "30"],
    "agama":     ["n01687978", "42"]
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Configuration selection for OOD study")
    parser.add_argument("--dataset", 
                        type = str, 
                        nargs="+",
                        choices=["ImageNetA_small", "ImageNetSD_small", "ImageNetVal_small",  "SignalBlur_small"], 
                        default=["ImageNetA_small"], 
                        help="Dataset selection")
    
    parser.add_argument("--model", 
                        type= str,
                        nargs="+",
                        choices=["ResNet18", "ViT"],
                        default=["ResNet18"], 
                        help="Model selection")
    
    parser.add_argument("--classes",
                        type =str, 
                        nargs="+", 
                        choices=class2id.keys(), 
                        default=["stingray", "junco", "bullfrog", "agama", "bad_eagle"], 
                        help="Class selection")
    
    parser.add_argument("--N", 
                        type= int,
                        default= 35,
                        help= "Number of images per class selection" )
  
    parser.add_argument("--resolution",
                        type = int,
                        default = 15,
                        help = "Resolution to generate a plane dataset given a triplet of images")
    
    #parser.add_argument("--loadDataset",
                        #type= str,
                        #default = "False",
                        #help= "Boolean parameter to download the dataset from Google Drive")
    
    parser.add_argument("--useFilteredPaths",
                        type= str,
                        default = "True",
                        help= "Boolean parameter to enforce the presence of correct predictions")
    
    args = parser.parse_args()
       
    if len(args.classes)<3:
        raise ValueError("Number of classes should be at least 3") 
    return args

#@profile
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print("Class selection:")
    class_selection = []
    for class_ in args.classes:
        class_selection.append(class2id[class_][0])
        print("\t", class_, class2id[class_])
    
    class_selection = sorted(class_selection)     
    
    # Create folder to store results
    create_result_folder(args.model) 
    
    # run experiment for the given configuration
    for dataset in args.dataset:
        for model in args.model:
            print(f"Dataset: \n\t{dataset}")
            print(f"Model: \n\t{model}")
            config = Configuration(model= model, 
                                   N = args.N, 
                                   dataset = dataset,
                                   id_classes= class_selection,
                                   resolution= args.resolution,
                                   useFilteredPaths = args.useFilteredPaths, 
                                   )
             
            # generate all the possible combinations between clasess and triplets of images
            filenamesCombis =  getCombiFromDBoptimalGoogleDrive(config)
            
            # modificar aqui para volver a runear
            tuneCombis = False
            endPoint = 10000  #10000
            if tuneCombis== True:
                randon.seed(42)
                mixCombis = filenamesCombis.copy()
                random.shuffle(filenamesCombis)
                mixCombis = mixCombis[:endPoint]
                
        
            # Create csv files to write results 
            # ---->>>>> he cambiado el nombre del file para second run
            with open(f"results/{model}/margin_values/{model}_{dataset}_N{config.N}_R{config.resolution}_run_2.csv", "w") as margin_csvfile:
                with open(f"results/{model}/classPredictions/{model}_{dataset}_N{config.N}_R{config.resolution}_run_2.csv", "w") as preds_csvfile:
                    for i, pathImgs in tqdm(enumerate(filenamesCombis), total=len(filenamesCombis), desc="Processing"):
                        #1. generate triplet object
                        triplet_obj= Triplet(pathImgs, config)
                        #2. create plane set 
                        planeset_obj = Planeset(triplet_obj, config)
                        #store triplet label, triplet predictions and unique classes predicted in planeset
                        preds_csvfile.write(f"{','.join(map(str, planeset_obj.triplet.true_label))},{','.join(map(str, planeset_obj.triplet.prediction))},{','.join(map(str, planeset_obj.predictedClasses))}\n")
                        #3. extract the margin for those images in the triplet correctly predicted
                        for j in range(3):
                            if triplet_obj.true_label[j] == triplet_obj.prediction[j]:
                                try:
                                    margin = round(PlanesetInfoExtractor(planeset_obj,config).margin[j],4)
                                except IndexError as e:
                                    print(f"IndexError: {e} at j={j}. Skipping this iteration.")
                                    margin = -1
                                    margin_csvfile.write(f"Error,{margin}\n")
                                    del margin
                                    continue
                                
                                #write results
                                margin_csvfile.write(f"{triplet_obj.true_label[j]},{margin}\n")
                                del margin
                        
                        # delete variables to free up memory
                        del triplet_obj
                        del planeset_obj
            
            del filenamesCombis
    print('\n All done. Results saved at results/')

if __name__ == "__main__":
    main()          
















