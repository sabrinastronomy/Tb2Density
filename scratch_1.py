import matplotlib.pyplot as plt
import numpy as np

samples = np.load("/Users/sabrinab/plots_num_samples_10000_eps_0_dvals_128_sigmaT_1_z_8_UNCORR_DENSITIES_SAMPLES.npy")

z = 8
actual_densities = np.load("/Users/sabrinab/actual_densities.npy")
size = len(actual_densities)

temp_bright = np.load("/Users/sabrinab/new_data/temp_bright.npy")
neutral = temp_bright > 0
print(neutral)
fig = plt.figure()
ax = fig.add_subplot(111)
for samp in samples[:100]:
    ax.plot(samp, alpha=1, c = 'blue')

xvals = np.linspace(0, size-1, size)
### PLOT actual BUBBLES
i_prev = 0
for i, bool in enumerate(neutral):
    if bool:
        if i_prev == 0 and i > 0:
            ax.plot(xvals[i_prev:i], actual_densities[i_prev:i], alpha=1, c='orange', label="Ionized", zorder=10)
        else:
            ax.plot(xvals[i_prev:i], actual_densities[i_prev:i], alpha=1, c='orange', zorder=10)
        i_prev = i
ax.plot(xvals, actual_densities, alpha = 1, c='k', label="Actual Densities")
ax.legend(prop={'size': 6})


### stolen aspect ratio code
ratio = 0.2
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
# the abs method is used to make sure that all numbers are positive
# because x and y axis of an axes maybe inversed.
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
####

fig.savefig("/Users/sabrinab/UNCORR_10000_samples_overview.pdf")
plt.show()
plt.clf()

import corner
test = samples[-1000:, -20:]
print(test.shape)
print(actual_densities[-5:])
figure = corner.corner(test, show_titles=True, labels=xvals, truths=actual_densities[-5:])
figure.savefig("/Users/sabrinab/UNCORR_corner_last_20.pdf")
