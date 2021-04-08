#### Setting default model, save, watershed directories
model_direc = "../MODELS/"
save_direc = '../STAT_IMAGE/'
watershed_direc = '/Users/sabrinaberger/CosmicDawn/watershed/analyze'
####

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import os
import imageio
import sys
sys.path.append('/Users/sabrinaberger/CosmicDawn/building/new_pix2pix/src/utils/')
import data_utils


class Model:
    def __init__(self, typ, filename, model_direc=model_direc, save_direc=save_direc, watershed_direc=watershed_direc, Rarr=np.arange(5, 25, 5), input_loc="/Users/sabrinaberger/CosmicDawn/building/21CMMC_Boxes_8/"):
        #### Essential folders for saving and opening files
        self.direc = save_direc # directory to save all new files created
        self.model_direc = model_direc # directory where model is stored
        self.watershed_direc = watershed_direc # directory where watershed code is stored
        ####

        #### Type of GAN (conditional, normal) and filename of model
        self.typ = typ
        self.filename = filename
        self.input_loc = input_loc
        ####

        #### Loading model
        self.loaded_model = load_model(self.model_direc + filename)
        ####

        self.Rarr = Rarr # radii of 2D histogram

        if typ == "gan":
            latent_dim = 100
            r, c = 5, 5
            self.input = np.random.normal(0, 1, (r * c, latent_dim))
            self.gen_imgs = self.loaded_model.predict(self.input)
        elif typ == "cgan":
            # Load and rescale data
            X_full_train, X_sketch_train, X_names = data_utils.load_data(test=False, data_dir=input_loc)
            X_full_train, X_sketch_train = np.expand_dims(X_full_train, axis=3), np.expand_dims(X_sketch_train, axis=3)

            X_full_val, X_sketch_val, X_names = data_utils.load_data(test=True, data_dir=input_loc)
            X_full_val, X_sketch_val = np.expand_dims(X_full_val, axis=3), np.expand_dims(X_sketch_val, axis=3)

            self.input = np.array(X_sketch_val)
            self.gen_imgs = self.loaded_model.predict(self.input)
            self.gen_imgs = np.asarray(self.gen_imgs)

        self.arr = self.gen_imgs[0, :, :, 0]    # testing arr (one slice of generated images to be used in 2D histogram)
        self.val_arr = X_full_val[0, :, :, 0]
        self.size = np.shape(self.arr)[0]

    def normal(self, x):
        return (x-np.min(x))/(np.max(x)-np.min(x))

    # Correlated histograms
    def plot_two_point_hist(self, name, tol=1, log=True):  # 0 to 10 or 0 to 5 convergence?
        # only use name if you're need_fig is set to False and you're not animating
        arr = self.arr
        R = self.R
        x1s = []
        x1_hists = []
        for i in range(np.shape(arr)[0]):
            for j in range(np.shape(arr)[1]):
                x1_hist = self.get_circumference([i, j], tol)
                x1s += len(x1_hist) * [arr[i,j]]
                x1_hists.extend(x1_hist)
        if log:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            H = plt.hist2d(x1s, x1_hists, norm=matplotlib.colors.LogNorm())
            plt.colorbar()
            ax.set_title("2-Point Correlation Histogram for R = {}".format(R))
        else:
            plt.hist2d(x1s, x1_hists)
            plt.colorbar()
            plt.title("2-Point Correlation Histogram for R = {}".format(R))

        plt.savefig(self.direc + "2point_{}_{}.png".format(R, name))

    def get_circumference(self, coords, tol):
        arr = self.arr
        R = self.R
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

        box = arr[minus_0:plus_0, minus_1:plus_1]  # Get square box of side length R around coords in arr
        x1_hist = []

        lim_0_box = np.shape(box)[0]
        lim_1_box = np.shape(box)[1]

        for i in range(lim_0_box):
            for j in range(lim_1_box):
                val = ((i - R_0) ** 2 + (j - R_1) ** 2) - (R ** 2)
                if tol >= val >= -tol:
                    x1_hist.append(box[i, j])
        return x1_hist


    def animate(self, name):
        arr = self.arr
        for R in self.Rarr:
            self.R = R # FIRST time defining R, changes R values within methods
            self.plot_two_point_hist(name)
        images = []
        for file_name in sorted(os.listdir(self.direc)):
            if file_name.endswith('.png'):
                file_path = os.path.join(self.direc, file_name)
                images.append(imageio.imread(file_path))
        imageio.mimsave(self.direc + "{}_movie.gif".format(name), images, duration=2)

    # Power Spectrum

    def twod_power_spectrum(self, data, delta, nbins):
        size = self.size
        fft_data = np.fft.fft2(np.fft.fftshift(data))
        power_data = np.abs(np.fft.fftshift(fft_data))**2

        kx = np.fft.fftfreq(size, delta)
        ky = np.fft.fftfreq(size, delta)

        k = []

        # Computing the square magnitude of each mode
        for i in range(len(kx)):
            for j in range(len(ky)):
                k.append(np.sqrt(kx[i]**2 + ky[j]**2))

        # DON'T UNDERSTAND EVERYTHING BELOW HERE
        # CREDIT: HANNAH

        hist, bin_edges = np.histogram(k, bins=nbins)

        a = np.zeros(len(bin_edges) - 1)  # here you need to take the number of BINS not bin edges!
        # you always need an extra edge than you have bin!

        c = np.zeros_like(a)
        kdelta = k[1] - k[0]
        # Here you sum all the pixels in each k bin.
        for i in range(size):
            for j in range(size):
                kmag = kdelta * np.sqrt(
                    (i - size / 2) ** 2 + (j - size / 2) ** 2)  # need to multiply by kdelta to get your k units
                for k in range(len(bin_edges)):  # make sure that you speed this up by not considering already binned ps's
                    if bin_edges[k] < kmag <= bin_edges[k + 1]:
                        a[k] += power_data[i, j]
                        c[k] += 1
                        break

        pk = (a / c) / ((delta * size) ** 2)
        kmodes = bin_edges[1:]
        return kmodes, pk

    def get_bubbles(self):
        np.save(self.gen_imgs)
        # python concavesitk.py