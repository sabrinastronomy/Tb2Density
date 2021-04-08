direc = "/Users/sabrinaberger/CosmicDawn/GANning/"
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# *uncomment if loading model for first time*
from keras.models import load_model

import preproc
import os
import imageio

png_dir = '/Users/sabrinaberger/CosmicDawn/GANning/png/'

# *uncomment if loading model for first time*
model = load_model(direc + 'GAN_xh1_4000.h5')
latent_dim = 100
r, c = 5, 5
noise = np.random.normal(0, 1, (r * c, latent_dim))
gen_imgs = model.predict(noise)

def normal(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

gen_imgs = np.save("gen_imgs.npy", gen_imgs)

plt.close()
plt.imshow(gen_imgs[0, :, :, 0])
plt.show()

X_train = preproc.load_data()

# Normal Histograms

gen_data = gen_imgs[:, :, :, 0].ravel()
# print("max_gen: {}".format(max(gen_data)))
# print("min_gen: {}".format(min(gen_data)))
# gen_data = normal(gen_data)

data = X_train[:, :, :].ravel()

print("average_gen: {}".format(np.average(gen_data)))
print("average_21: {}".format(np.average(data)))
print("max_gen: {}".format(max(gen_data)))
print("min_gen: {}".format(min(gen_data)))

binwidth = 0.01
plt.close()
#
# plt.hist(gen_data, bins=np.arange(min(gen_data), max(gen_data) + binwidth, binwidth), alpha=0.3, density=True,
#          label="Generated Data")
# plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), color='k', alpha=0.3, density=True,
#          label="Original Data")
# plt.xlabel("$X_{HI}$")
# plt.title("Generated and Original Ionization Maps Histogram (softmax layer)")
# plt.legend()
# plt.savefig('gen_sm.png')
# plt.close()

# Correlated histograms
def plot_two_point_hist(arr, name, R=10, tol=1, log=True, dir=png_dir):  # 0 to 10 or 0 to 5 convergence?
    # only use name if you're need_fig is set to False and you're not animating

    x1s = []
    x1_hists = []

    for i in range(np.shape(arr)[0]):
        for j in range(np.shape(arr)[1]):
            x1_hist = get_circumference([i, j], arr, R, tol)

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

    plt.savefig(dir + "2point_{}_{}.png".format(R, name))

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


def animate(arr, name, Rarr, dir):
    for R in Rarr:
        plot_two_point_hist(arr, name, R, dir=dir)
    images = []
    for file_name in os.listdir(dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(dir + "{}_movie.gif".format(name), images)


# animate(X_train[0], "21cmFAST Data (log)", np.arange(20, 40, 2),  dir=png_dir + "21cm/")
# animate(gen_imgs[0, :, :, 0], "GAN Data (log)", np.arange(10, 30, 2), dir=png_dir + "GAN/")


# plot_two_point_hist(gen_imgs[0, :, :, 0], name="GAN Data (log) (softmax layer)", log=True)
# plot_two_point_hist(X_train[0], name="21cmFAST Data (log) (softmax layer)", log=True)

# norm_x = normal(gen_imgs[0, :, :, 0])
# plot_two_point_hist(norm_x, name="Scaled GAN Data (log) (softmax layer)", log=True)



# Power Spectrum

def twod_power_spectrum(data, size, delta, nbins):
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
    # you alwaysneed an extra edge than you have bin!

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

plt.close()
kmodes, pk = twod_power_spectrum(gen_imgs[0], 200, 1, 20)
o_kmodes, o_pk = twod_power_spectrum(X_train[0], 200, 1, 20)
plt.plot(kmodes, kmodes**2 * pk, label="Sample Generated Data")
plt.title("Rough: Generated Data Power Spectra")
plt.xlabel("k")
plt.ylabel("$k^{2}P(k)$")
plt.tight_layout()
plt.savefig("gen_PS.png")
plt.close()

plt.xlabel("k")
plt.ylabel("$k^{2}P(k)$")
plt.plot(o_kmodes, kmodes**2 * o_pk, label="Sample Original Data")
plt.title("Rough: Original Data Power Spectra")
plt.tight_layout()

plt.savefig("og_PS.png")