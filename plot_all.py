import matplotlib.pyplot as plt
import numpy as np
import sys
direc = sys.argv[1]
print(direc)
redshifts = [10, 8, 6, 4]
for redshift in redshifts:
    corr_samples = np.load("{}/plots_num_samples_10000_eps_0_dvals_128_sigmaT_1_z_{}_CORRELATED_DENSITIES_{}_SAMPLES.npy".format(direc, redshift, redshift))
    uncorr_samples = np.load("{}/plots_num_samples_10000_eps_0_dvals_128_sigmaT_1_z_{}_UNCORR_DENSITIES_{}_SAMPLES.npy".format(direc, redshift, redshift))
    for samples, type in zip([corr_samples, uncorr_samples], ["CORR", "UNCORR"]):
        actual_densities = np.load("{}/actual_densities_{}.npy".format(direc, redshift))
        size = len(actual_densities)
        print(actual_densities[-5:])
        truths_data = actual_densities[-20:]
        temp_bright = np.load("{}/temp_bright_{}.npy".format(direc, redshift))
        neutral = temp_bright > 0
        print(neutral)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for samp in samples[:1000]:
            ax.plot(samp, alpha=0.1, c = 'blue')

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

        fig.savefig("{}/{}_{}_10000_samples_overview.pdf".format(direc, redshift, type))
        plt.clf()

        import corner
        test = samples[-1000:, -20:]
        test_length = len(test)
        figure = corner.corner(test, show_titles=True, labels=xvals, truths=truths_data)
        figure.savefig("{}/{}_corner_last_{}.pdf".format(direc, type, redshift))
