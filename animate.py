import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from wand.image import Image


def get_circumference(coords, arr, R, tol):
    plus_0 = R + coords[0]
    minus_0 = -R + coords[0]

    plus_1 = R + coords[1]
    minus_1 = -R + coords[1]

    lim_0 = np.shape(arr)[0]
    lim_1 = np.shape(arr)[1]

    R_0 = R_1 = R
    arr_center = arr[coords[0], coords[1]]

    if plus_0 >= lim_0 - 1:
        plus_0 = lim_0
        R_0 = R
    if plus_1 >= lim_1 - 1:
        plus_1 = lim_1
        R_1 = R
    if minus_1 < 0:
        minus_1 = 0
        R_1 = minus_1
    if minus_0 < 0:
        minus_0 = 0
        R_0 = 0

    # x_coords = []
    # y_coords = []

    box = arr[minus_0:plus_0, minus_1:plus_1]  # Get square box of side length R around coords in arr
    x1_hist = []

    # box_center = box[R_0, R_1]

    lim_0_box = np.shape(box)[0]
    lim_1_box = np.shape(box)[1]

    for i in range(lim_0_box):
        for j in range(lim_1_box):
            val = ((i - R_0) ** 2 + (j - R_1) ** 2) - (R ** 2)
            if tol >= val >= -tol:
                x1_hist.append(box[i, j])
    return x1_hist


def animate(name, direc):
    images = [] 
    
    # Convert .pdf images to .png
    # for file_name in os.listdir(direc):
    #     with Image(filename=direc + file_name) as i:
    #         i.format = 'png'
    #         i.save(filename=direc + file_name.split(".")[0] + ".png")

    for file_name in os.listdir(direc):
        if file_name.endswith('.png'):
            file_path = os.path.join(direc, file_name)
            images.append(imageio.imread(file_path))
    print(images)
    imageio.mimsave(direc + "{}_movie.gif".format(name), images)
    
animate("train", "/Users/sabrinaberger/CosmicDawn/plots_epochs_20000/train/")
animate("val", "/Users/sabrinaberger/CosmicDawn/plots_epochs_20000/val/")



