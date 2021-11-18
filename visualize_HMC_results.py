import pymc3 as pm
import numpy as np
from battaglia_full import Dens2bBatt
import h5py
import glob
import matplotlib.pyplot as plt
import corner
z = 9
one_d_size = 16
trace_name = "z_{}_{}.trace".format(z, one_d_size)
path_CORR_DATA = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/CORR_DATA/"
path_TRACES = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/TRACES/"
init_samples = np.load("/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/CORR_DATA/inital_{}_{}.npy".format(z, one_d_size))
print(init_samples)
# dir = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/21cmFASTData/21cmFASTBoxes_{}/PerturbedField_*".format(z)
# hf = h5py.File(glob.glob(dir)[0], 'r')
# line_rho = hf["PerturbedField/density"][:, 0, 0]
# dens2Tb = Dens2bBatt(line_rho, 1, z, one_d=True)
# line_Tb = one_Tb = dens2Tb.temp_brightness
# one_rho = actual_rho = line_rho[:one_d_size]
# one_Tb = actual_Tb = line_Tb[:one_d_size]


one_rho = np.load(path_CORR_DATA + "one_rho_z{}_size{}.npy".format(z, one_d_size))
one_Tb = np.load(path_CORR_DATA + "one_Tb_z{}_size{}.npy".format(z, one_d_size))
print(one_Tb)
samples_pymc3 = []
labels = []

with pm.Model():
    a = pm.load_trace(path_TRACES + trace_name)
    last_samples = []
    for i in range(one_d_size):
        last_samples.append(a.get_values('d_{}'.format(i)))
        print(len(a.get_values('d_{}'.format(i))))
    # for i in range(one_d_size):
    #     print(len(a.get_values('d_{}'.format(i))))
    #     samples_pymc3.append(a.get_values('d_{}'.format(i)))
    #     labels.append('d_{}'.format(i))

# stacked_samples = np.vstack((samples_pymc3)).T

# fig = corner.corner(stacked_samples, labels=labels, truths=one_rho)
# plt.savefig("corner_9_all.png", dpi=200)
# plt.close()

ionized = np.where(one_Tb == 0)[0]
print("ionized pixels at {}".format(ionized))
for i in ionized:
    plt.scatter(i, one_rho[i], c="k", zorder=100, s=10)
plt.scatter(i, one_rho[i], label="ionized pixel", c="k", zorder=100, s=10)

print(init_samples)
plt.plot(init_samples, label="initial samples", alpha=1, c="orange", ls="--", zorder=100)
plt.plot(one_rho, label="actual densities", alpha=0.5, c="purple", ls="--")

trans_s = np.transpose(last_samples)[-100:]
for last in trans_s:
    plt.plot(last, alpha=0.5, c="pink")
plt.plot(last, alpha=0.5, c="pink", label="last 100 HMC samples")
plt.xlim((13, 14.5))
plt.xlabel("pixel value")
plt.ylabel("overdensity")
plt.title("21cmFAST Density Distribution at z = {}".format(z))
plt.legend()
plt.savefig("yayz{}_size{}.png".format(z, one_d_size), dpi=200)
