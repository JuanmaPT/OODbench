#generate a color map for ImageNet1k 
import matplotlib.pyplot as plt
import numpy as np

def generate_color_map():
    # Condisering different color maps from matplotlib till generate 1000 different colors
    color_list = []
    colors_reached = False
    colormaps = [
        plt.cm.rainbow,
        plt.cm.plasma,
        plt.cm.viridis,
        plt.cm.twilight,
        plt.cm.twilight_shifted,
        plt.cm.gnuplot,
        plt.cm.gnuplot2,
        plt.cm.nipy_spectral,
    ]

    for j,cmap in enumerate(colormaps):
        print(j)
        for i in range(cmap.N):
            color = np.array(cmap(i / cmap.N))  # Normalize index to [0, 1]
            if not any(np.allclose(color, c, rtol=1e-30, atol=1e-30) for c in color_list):
                color_list.append(color)
            if len(color_list) == 1000:   
                colors_reached = True
                break
        if colors_reached:
            break

    # Shuffle the entire color list
    np.random.seed(26)
    np.random.shuffle(color_list)
    return color_list


def display_colors(color_list):
    fig, ax = plt.subplots(figsize=(15, 1))

    # Plot color bars for each color in the list
    for i, color in enumerate(color_list):
        ax.fill_between([i, i + 1], 0, 1, color=color)

    # Remove y-axis ticks and labels
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Set the x-axis ticks at multiples of 100
    ax.set_xticks(range(0, 1001, 100))
    ax.set_xticklabels(range(0, 1001, 100))

   
    ax.set_title("Color Map")
    plt.show()





colors = generate_color_map()
display_colors(colors)
np.savetxt('Colors1k.txt', colors)
