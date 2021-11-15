import pymc3 as pm
import numpy as np
from battaglia_full import Dens2bBatt
import h5py
import glob
import matplotlib.pyplot as plt
import corner
z = 9
one_d_size = 64
trace_name = "z_{}_{}.trace".format(z, one_d_size)
path_CORR_DATA = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/CORR_DATA/"
path_TRACES = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/TRACES/"

# dir = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/21cmFASTData/21cmFASTBoxes_{}/PerturbedField_*".format(z)
# hf = h5py.File(glob.glob(dir)[0], 'r')
# line_rho = hf["PerturbedField/density"][:, 0, 0]
# dens2Tb = Dens2bBatt(line_rho, 1, z, one_d=True)
# line_Tb = one_Tb = dens2Tb.temp_brightness
# one_rho = actual_rho = line_rho[:one_d_size]
# one_Tb = actual_Tb = line_Tb[:one_d_size]

# one_rho = [ 0.24468348, -0.65290994, -0.81941303,  0.25609795,  0.11437628,  2.94068976,
#   1.70903666,  1.85774218]
# Tb = [ 33.60645392   9.37143174   4.87584829  33.91464473  30.08815965
#  106.39862347  73.14398985  77.15903891]

one_rho = np.load(path_CORR_DATA + "one_rho_z{}_size64.npy".format(z))[:one_d_size]
one_Tb = np.load(path_CORR_DATA + "one_Tb_z{}_size64.npy".format(z))[:one_d_size]
print(one_Tb)
samples_pymc3 = []
labels = []
print(max(one_rho))

with pm.Model():
    a = pm.load_trace(path_TRACES + trace_name)
    last_samples = []
    for i in range(one_d_size):
        last_samples.append(a.get_values('d_{}'.format(i)))
    for i in range(one_d_size):
        print(len(a.get_values('d_{}'.format(i))))
        samples_pymc3.append(a.get_values('d_{}'.format(i)))
        labels.append('d_{}'.format(i))

stacked_samples = np.vstack((samples_pymc3[16:24])).T

# fig = corner.corner(stacked_samples, labels=labels[16:24], truths=one_rho[16:24])
# plt.savefig("corner_ionized_all.png", dpi=200)
# plt.close()

ionized = np.where(one_Tb == 0)[0]
for i in ionized:
    plt.scatter(i, one_rho[i], c="k", zorder=100, s=10)
plt.scatter(i, one_rho[i], label="ionized pixel", c="k", zorder=100, s=10)

plt.plot(one_rho, label="actual densities", alpha=0.5, c="purple", zorder=90, ls="--")

trans_s = np.transpose(last_samples)
for last in trans_s:
    plt.plot(last, alpha=0.5, c="turquoise")

plt.plot(last, alpha=0.5, c="turquoise", label="HMC samples")

plt.xlabel("pixel value")
plt.ylabel("overdensity")
plt.title("21cmFAST Density Distribution at z = {}, 100 samples (short run)".format(z))
plt.legend()
plt.savefig("yay{}.png".format(z), dpi=200)
