import itertools
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import pickle

def imgToTensor(img):
    #resize = transforms.Resize([size, size]) # note images have different size
    to_tensor = transforms.ToTensor()
    tensorImg= to_tensor(img)
    #tensorImg= tensorImg.unsqueeze(0) #comment this if we want (3,3)
    return tensorImg

def margin_TRDP_I (class_pred,pred_matrix,idx_pred_im,ground_truth_im,accuracy_triplet,margin_triplet):
    accuracy_row = []
    margin_row = []
    for i in range(3):
        accuracy_row.append(class_pred[idx_pred_im[i]] == ground_truth_im[i])
        margin_row.append(margin_of_image(idx2label_mat(idx_pred_im[i]), pred_matrix, ground_truth_im[i]))
    print(accuracy_row)
    print(margin_row)
    accuracy_triplet.append(accuracy_row)
    margin_triplet.append(margin_row)
    return accuracy_triplet, margin_triplet

def save_results(results_dir,results):
    # Save the results dictionary to a file in the results folder
    results_file = os.path.join(results_dir, "results.pkl")

    # You can use the "pickle" library to save and load Python objects
    import pickle

    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to {results_file}")



def getCombiFromDB(c1, c2, c3,db):

    filenames_combinations = []

    # Get the file paths of the images in each folder
    class1_folder= db+ c1 +"/"
    class2_folder= db+ c2 + "/"
    class3_folder= db+ c3 + "/"
   
    class1 = [os.path.join(class1_folder, filename) for filename in os.listdir(class1_folder)]
    class2 = [os.path.join(class2_folder, filename) for filename in os.listdir(class2_folder)]
    class3 = [os.path.join(class3_folder, filename) for filename in os.listdir(class3_folder)]
   
    #get all possible combinations
    combinations = list(itertools.product(class1, class2, class3))
   
    imgCombinationsTensor =[]    
    for combi in combinations:
        img1= imgToTensor(Image.open(combi[0]))
        img2= imgToTensor(Image.open(combi[1]))
        img3= imgToTensor(Image.open(combi[2]))

        imgCombinationsTensor.append(combi)
       
        imgCombinationsTensor.append([img1, img2, img3])
   
    return imgCombinationsTensor, filenames_combinations

def extract_idx_triplets_from_tensor(images,class_pred,planeloader):
        distances = []  # List to store the distances
        triplet_index = []
        for image_idx in range(3):
            for batch_idx, inputs in enumerate(planeloader):
                distance = torch.dist(inputs, images[image_idx])
                distances.append(distance.item()) 
            min_distance_index = distances.index(min(distances))
            # Print the result
            print("Index of the minimum distance:", min_distance_index)
            # Mapping the class predictions to a range of 0 to 9
            distances = [] 
            triplet_index.append(min_distance_index)
        return triplet_index

def mean_distance_to_class_label(anchor, array, class_label):
    distances = []
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            if array[row, col] == class_label:
                position = np.array([row, col])
                distance = np.linalg.norm(anchor - position)
                distances.append(distance)
    mean_distance = np.mean(distances)
    return mean_distance

def margin_of_image(anchor, array, class_label):
    distances = []
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            if array[row, col] == class_label:
                position = np.array([row, col])
                distance = np.linalg.norm(anchor - position)
                distances.append(distance)
    mean_distance = np.mean(distances)
    adjusted_distances = distances - mean_distance
    try:
        margin = np.max(adjusted_distances)
    except ValueError:
        margin = np.nan
    return margin

def idx2label_mat(list_index):
    row = list_index // 100
    column = list_index % 100
    return row, column

def computeMargin(imgIdx,   ):
    #create the all idx
    allIdx = np.asarray(list(range(10001)),dtype=int)
    #get the indexes for the samples belonging to the class
    classIdx = np.asarray([allIdx[i] for i in range(len(class_pred)) if class_pred[i] == class_label])
    #get mean and distances from our image to every image belonging to class
    meanClass= np.mean(classIdx)
    dClass= np.abs(imgIdx -classIdx)



    #get the margin 
    margin = np.max(np.where(meanClass > dClass, meanClass, dClass)) #elements from mean when mean > distance

    return margin

def mapCoordToIdx(coord):
        # return the closest position to the seed    
    return 800







def save_preds_to_file(n_combis, generate_matrix_func, results_dir="results"):
    results = {}  # Initialize an empty dictionary to store matrices

    for i_triplet in range(n_combis):
        pred_matrix = generate_matrix_func()  # Call the provided function to generate pred_matrix
        
        # Store the pred_matrix in the results dictionary
        results[f"Matrix_{i_triplet}"] = pred_matrix

    # Create the directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the results dictionary to a file in the results folder
    results_file = os.path.join(results_dir, "results.pkl")

    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to {results_file}")

# Example function to generate pred_matrix
def generate_matrix():
    return np.random.randint(0, 1001, size=(10, 10))

# Example usage of the function
n_combis = 10  # Replace with the desired number of combinations
results_directory = "results"  # Replace with the desired results directory
save_matrices_to_file(n_combis, generate_matrix, results_directory)
