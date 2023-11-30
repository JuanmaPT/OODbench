import pickle

# Load variables from the file
with open('variables.pkl', 'rb') as file:
    data = pickle.load(file)

planeset = data['planeset']
cmap = data['cmap']
color_dict = data['color_dict']

planeset.show_simple(cmap,color_dict)
