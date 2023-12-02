from TripletInfoExtractor import *
import pickle

path_results = './results/results_3_3_imagenet_original.pkl'
with open(path_results, 'rb') as file:
    data = pickle.load(file)

matrix = data['Matrix_0']

anchors = {235:(4, 4), 562:(44, 4), 609:(20, 44)}
tie = TripletInfoExtractor(matrix, anchors)
region_max_distances = tie.get_max_distance_transforms()
anchor_distances= tie.get_distance_from_anchor_to_border()
distances_class3= tie.get_distances_and_orientations()[2]
plt.imshow(tie.connected_components[2])
regProp= tie.get_RegionProps()
margins= tie.get_margins()
dts = tie.distance_transforms


components= tie.connected_components
fig, ax = plt.subplots()
img = ax.imshow(dts[1])
plt.show()

print(len(components[0]))