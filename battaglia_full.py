import matplotlib.pyplot as plt
import numpy as np
import itertools
import h5py
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Toy Density Field
nx, ny = (5, 5)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)

# TESTING: density3D = np.random.normal(0, 0.1, (64, 64, 64)) # numpy.random.normal(loc=0.0, scale=1.0, size=None)
density1D = np.random.normal(0, 1, 64) # numpy.random.normal(loc=0.0, scale=1.0, size=None)

class Dens2bBatt:
    """
    This class follows the Battaglia et al (2013) model to go from a density field to a temperature brightness field.
    """
    def __init__(self, density, delta_pos, set_z, one_d=True, flow=True):
        # go into k-space
        self.one_d = one_d
        self.density = density
        self.set_z = set_z

        self.delta_pos = delta_pos  # Mpc
        if one_d:
            self.cube_len = len(self.density)
            self.integrand = self.density * self.delta_pos  # weird FFT scaling for 1D
            self.ks = np.fft.ifftshift(np.fft.fftfreq(len(self.density), self.delta_pos))
            self.k_mags = np.abs(self.ks)
            self.X_HI = np.empty(self.cube_len)
            self.delta_k = self.ks[1] - self.ks[0] 

        else: # assuming 3D
            self.cube_len = len(self.density[:, 0, 0])
            self.integrand = self.density * (self.delta_pos**3) # weird FFT scaling for 3D
            self.kx = np.fft.ifftshift(np.fft.fftfreq(self.density.shape[0], self.delta_pos))
            self.ky = np.fft.ifftshift(np.fft.fftfreq(self.density.shape[1], self.delta_pos))
            self.kz = np.fft.ifftshift(np.fft.fftfreq(self.density.shape[2], self.delta_pos))

            self.kx *= 2 * np.pi  # scaling k modes correctly
            self.ky *= 2 * np.pi  # scaling k modes correctly
            self.kz *= 2 * np.pi  # scaling k modes correctly

            self.k_mags = np.sqrt(self.kx ** 2 + self.ky ** 2 + self.kz ** 2)
            self.X_HI = np.empty((self.cube_len, self.cube_len, self.cube_len))
            self.delta_k = self.kx[1] - self.kx[0]  
            self.xdim = self.ydim = self.zdim = self.cube_len

        self.density_k = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(self.integrand)))

        self.tophatted_ks = []
        self.get_tophatted_ks()

        self.rs_top_hat_3d = lambda k: 3 * (np.sin(k) - np.cos(k) * k) / k ** 3  # pixelate and smoothing? 
        self.rs_top_hat_1d = lambda arg: np.sinc(arg / np.pi)  # engineer's sinc so divide argument by pi

        self.z_re = 0
        self.delta_z = 0
        self.temp_brightness = 0
        self.avg_z = 8

        self.b_0 = 0.593
        self.alpha = 0.564
        self.k_0 = 0.185
        self.b_mz = lambda k: self.b_0 / (1 + k / self.k_0) ** self.alpha # bias factor (8) in paper

        if flow:
            self.flow()

    def get_tophatted_ks(self):
        for k in self.k_mags:
            if self.one_d:
                self.tophatted_ks.append(np.sinc(k / np.pi))
            elif k > 1e-6:
                self.tophatted_ks.append(3 * (np.sin(k) - np.cos(k) * k) / k ** 3)
            else:
                self.tophatted_ks.append(1 - k ** 2 / 10) # taylor expansion of tophat function TODO: derive this
        return self.tophatted_ks

    def apply_filter(self):
        w_z = lambda k: (self.b_mz(k) * self.tophatted_ks)  # filter (10) in paper

        self.density_k = self.density_k * w_z(self.k_mags*self.delta_pos)
        if self.one_d:
            self.density_k *= self.delta_k
        else:
            self.density_k *= self.delta_k**3 # scaling amplitude in fourier space for 3D
        self.delta_z = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(self.density_k)))
        if self.one_d:
            self.delta_z *= self.cube_len / (2*np.pi)
        else:
            self.delta_z *= (self.cube_len**3)/(2*np.pi)**3 # weird FFT scaling for 3D, getting rid of 1/n^3
        # recover delta_z field
        return self.delta_z

    def get_z_re(self): # average z: order of half ionized half neutral
        self.z_re = self.delta_z * (1 + self.avg_z) + (1 + self.avg_z) - 1
        return self.z_re

    def get_x_hi(self):
        if self.one_d:
            i = 0
            for z in self.z_re:
                if z < self.set_z:
                    self.X_HI[i] = 1
                else:
                    self.X_HI[i] = 0
                i += 1
        else:
            for x, y, z in itertools.product(*map(range, (self.xdim, self.ydim, self.zdim))): 
                if self.z_re[x, y, z] < self.set_z:
                    self.X_HI[x, y, z] = 1
                else:
                    self.X_HI[x, y, z] = 0
        return self.X_HI

    def get_temp_brightness(self):
        first = 27 * self.X_HI
        second = 1 + self.density
        self.temp_brightness = first*second
        return self.temp_brightness

    def flow(self):
        self.apply_filter()
        self.get_z_re()
        self.get_x_hi()
        self.get_temp_brightness()
        # print("Temperature brightness field created from density field.")
        return



# if __name__ == "__main__":
    # rho = np.asarray([1.92688, -0.41562])
    # z = 15
    # dens2Tb = Dens2bBatt(rho, 1, z)
    # Tb = dens2Tb.temp_brightness
    # print(Tb)

#     for z in [14]:
#         dir = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/building/21cmFASTBoxes_{}/PerturbedField_*".format(z)
#         hf = h5py.File(glob.glob(dir)[0], 'r')
#         data = hf["PerturbedField/density"]
#
#         #### TO RUN A DENSITY FIELD TO TEMPERATURE BRIGHTNESS CONVERSION
#
#         density3D = np.asarray(data)
#         dens2Tb = Dens2bBatt(density3D, 1, z)
#         Tb = dens2Tb.temp_brightness
#         print(Tb)
#         z_re = dens2Tb.z_re
#
#         ####
#
#         slice_rho = density3D[:, :, 0]
#         slice_delta_z = dens2Tb.delta_z[:, :, 0]
#         slice_Tb = Tb[:, :, 0]
#
#         plt.close()
#         fig, (ax1, ax2) = plt.subplots(ncols=2)
#         im1 = ax1.imshow(slice_rho)
#         divider = make_axes_locatable(ax1)
#         cax1 = divider.append_axes("right", size="5%", pad=0.05)
#         fig.colorbar(im1, cax=cax1)
#         ax1.set_title(r"$\delta_{\rho}$" + ", z = {}".format(z))
#
#         im2 = ax2.imshow(slice_Tb)
#         divider = make_axes_locatable(ax2)
#         cax2 = divider.append_axes("right", size="5%", pad=0.05)
#         fig.colorbar(im2, cax=cax2)
#         ax2.set_title(r"$\delta T_b$" + ", z = {}".format(z))
#         plt.tight_layout(h_pad=1)
#         fig.savefig("uni_rho_T_b_{}.png".format(z))
#         plt.close()