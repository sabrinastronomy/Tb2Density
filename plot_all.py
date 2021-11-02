import matplotlib.pyplot as plt
import numpy as np
import sys
import corner
import triangle
import seaborn
import seaborn as sns
from battaglia_full import Dens2bBatt

# Applying the default theme
sns.set_theme()


def oned_power_spectrum(data, delta, nbins):
    size = len(data)
    input = data * delta # scaling amplitude, Mpc
    fft_data = np.fft.fftshift(np.fft.fft(input))
    # fft_data = np.fft.fft(input)
    power_data = np.abs(fft_data) ** 2
    # k_arr = np.abs(np.fft.fftshift(np.fft.fftfreq(size, delta) * 2 * np.pi))  # extra 2pi needed here
    k_arr = np.abs(np.fft.fftshift(np.fft.fftfreq(size, delta) * 2 * np.pi))  # extra 2pi needed here
    # hist, bin_edges = np.histogram(k_arr, bins=nbins) # use histogram to get bin edges given number of bins
    # sum_kmode = np.zeros(len(bin_edges) - 1)  # array to sum power in corresponding to each k mode
    # num_in_kmode = np.zeros_like(sum_kmode) # number of k modes in bin
    # kdelta = np.abs(k_arr[1] - k_arr[0])
    # for i in range(size):
    #     kmag = kdelta * np.abs((size/2)-i)  # getting k magnitude in terms of kdelta units
    #     for k in range(len(k_bin_edges) - 1):
    #         if k_bin_edges[k] < kmag <= k_bin_edges[k + 1]:
    #             sum_kmode[k] += power_data[i]
    #             num_in_kmode[k] += 1
    #             break
    # power_k = (sum_kmode / num_in_kmode) / (delta * size)
    # k_bin_edges = (k_arr[:-1] + k_arr[1:])/2
    half = int(size/2)
    return k_arr[half:], power_data[half:]

def get_neutral_ionized_boolean(temps):
    # returns ionized and neutral boolean areas
    neutral = temps > 0
    ionized = np.asarray([not el for el in neutral])
    return neutral, ionized

def plot_sigma_vs_z(errs, labels, redshifts, direc):
    # RESULT PLOT SHOWING SIGMA VS Z
    if len(errs) > 1:
        for err, lab in zip(errs, labels):
            plt.plot(redshifts, err, label=lab, alpha=0.8, marker='o')
        plt.legend()
        plt.xlabel("Redshift")
        plt.ylabel("Error")
        plt.xlim(redshifts[0], redshifts[-1])
        plt.savefig("{}/sigma_vs_z_{}.pdf".format(direc, labels), dpi=200)
        plt.clf()
        return
    else:
        plt.plot(errs, redshifts, label=labels)
        plt.xlim(redshifts[0], redshifts[-1])
        plt.savefig("{}/sigma_vs_z_{}.pdf".format(direc, labels), dpi=200)
        plt.clf()
        return

def get_percentile(arr, perc):
    # Takes in multidimensional array of samples with shape (dimensions, num_samples) and percentile value wanted
    percentiled_arr = []
    for pixel in arr:
        percentile_pixel = np.percentile(pixel, perc)
        percentiled_arr.append(percentile_pixel)
    return percentiled_arr


def plot_overview_68perc(samples, actual_densities, temps, type, direc, redshift, num_samples=1000):
    # RESULT PLOT SHOWING OVERVIEW OF PIXELS IN DENSITY FIELD (IONIZED/NEUTRAL)
    neutral, ionized = get_neutral_ionized_boolean(temps)
    print("At redshift {}, at least one neutral = {}".format(redshift, np.any(neutral==True)))
    size = len(actual_densities) # size of density field
    print("size {}".format(size))
    xvals = np.linspace(0, size-1, size)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    transposed_samples = np.transpose(samples)
    perc84_samps = get_percentile(transposed_samples, 84)
    perc16_samps = get_percentile(transposed_samples, 16)
    median_samps = get_percentile(transposed_samples, 50)

    ax.plot(xvals, perc84_samps, alpha = 0.7, c='b', ls="-", label="P84 Sampled Densities", zorder=10)
    ax.plot(xvals, perc16_samps, alpha = 0.7, c='b', ls="-", label="P16 Sampled Densities", zorder=10)
    ax.plot(xvals, median_samps, alpha = 0.5, c='aqua', ls="-", label="Median Sampled Densities")

    ax.plot(xvals, actual_densities, alpha = 0.5, c='k', ls="--", label="Truth Densities", zorder=10)
    ax.set_title("z = {}".format(redshift))
    ax.set_xlabel("Pixel Number", fontsize=5)
    ax.set_ylabel(r"Fractional overdensity $\delta$", fontsize=5)

    i = 0
    stopping = (len(actual_densities) - 1)
    first = True
    while i <= stopping:
        if ionized[i]:
            i_beginning = i
            while ionized[i] and i <= stopping and i < (len(actual_densities) - 1):
                i += 1
            i_last = i+1
            if first:
                ax.plot(xvals[i_beginning:i], actual_densities[i_beginning:i], alpha=1, c='gold', label="Ionized", zorder=100)
            else:
                ax.plot(xvals[i_beginning:i_last], actual_densities[i_beginning:i_last], alpha=1, c='gold', zorder=100)
            first = False
        i += 1

    ax.legend(prop={'size': 6})


    ## stolen aspect ratio code
    ratio = 0.2
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ###

    fig.savefig("{}/{}_{}_{}_densities_{}_samples_64CRED_overview.png".format(direc, redshift, type, size, num_samples), dpi=200, bbox_inches='tight')
    plt.clf()

def plot_overview(samples, actual_densities, temps, type, direc, redshift, num_samples=1000):
    # RESULT PLOT SHOWING OVERVIEW OF PIXELS IN DENSITY FIELD (IONIZED/NEUTRAL)
    neutral, ionized = get_neutral_ionized_boolean(temps)
    print("At redshift {}, at least one neutral = {}".format(redshift, np.any(neutral==True)))
    size = len(actual_densities) # size of density field
    print("size {}".format(size))
    xvals = np.linspace(0, size-1, size)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(np.shape(samples[-num_samples:]))
    for i, samp in enumerate(samples[-num_samples:]):
        if i == 0:
            ax.plot(xvals, samp, alpha=0.4, c = 'aqua', label="Sampled Densities")
        ax.plot(xvals, samp, alpha=0.4, c='aqua')
    ax.plot(xvals, actual_densities, alpha = 1, c='k', ls="--", label="Truth Densities", zorder=10)
    ax.set_title("z = {}".format(redshift))
    ax.set_xlabel("Pixel Number", fontsize=5)
    ax.set_ylabel(r"Fractional overdensity $\delta$", fontsize=5)
    # ax.set_ylim((0, 10))

    i = 0
    stopping = (len(actual_densities) - 1)
    first = True
    while i <= stopping:
        if ionized[i]:
            i_beginning = i
            while ionized[i] and i <= stopping and i < (len(actual_densities) - 1):
                i += 1
            i_last = i+1
            if first:
                ax.plot(xvals[i_beginning:i], actual_densities[i_beginning:i], alpha=1, c='gold', label="Ionized", zorder=100)
            else:
                ax.plot(xvals[i_beginning:i_last], actual_densities[i_beginning:i_last], alpha=1, c='gold', zorder=100)
            first = False
        i += 1

    ax.legend(prop={'size': 6})


    ## stolen aspect ratio code
    ratio = 0.2
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ###

    fig.savefig("{}/{}_{}_{}_densities_{}_samples_overview.png".format(direc, redshift, type, size, num_samples), dpi=200, bbox_inches='tight')
    plt.clf()

def get_power_specs(k_arr, power_arr, label_arr, direc, redshift, colors, ylimit=20):
    for ks, power, lab, color in zip(k_arr, power_arr, label_arr, colors):
        if lab == r"$P_{constructed}$":
            for pow in power:
                plt.plot(ks, pow, alpha=0.5, c=color, lw=0.2)
            plt.plot(ks, power[-1], label=lab, c=color, alpha=0.9, lw=0.2)
        else:
            plt.plot(ks, power, label=lab, c=color, alpha=0.8, ls="--", zorder=100)
        # print(ks)
    plt.xlabel("k")
    plt.ylabel("Power")
    plt.title("Power Spectra at z = {}".format(redshift))
    # plt.ylim((0, ylimit))
    plt.legend()
    plt.savefig("{}/pspec_{}.pdf".format(direc, redshift), dpi=400)
    plt.savefig("{}/pspec_{}.png".format(direc, redshift), dpi=200)
    plt.clf()

def get_1sigma(samples):
    return (np.percentile(samples, 84) - np.percentile(samples, 16))/2


direc = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA"
plot_direc = "/Users/sabrinaberger/debug_15"
redshifts = [12]
# redshifts = [10]
err_corr = []
err_uncorr = []
err_corr_sigma_neutral = []
err_corr_sigma_ionized = []


# get sigmas
perc = False
# get overview plot
over = True
# get 64 cred region
perc_68_over = False
# get power
power = False
nbins = 65
num_samples = 1000

#
# for redshift in [6,8]:
#     actual_densities = np.load("{}/actual_densities_old.npy".format(direc))
#     temp_bright = np.load("{}/temp_bright_{}.npy".format(direc, redshift))
#     # ks, true_pspec = oned_power_spectrum(actual_densities, 1, nbins)
#     corr_samples = np.load("../plots_num_samples_10000_eps_0_dvals_10_sigmaT_1_z_{}_CORRELATED_DENSITIES_{}_SAMPLES.npy".format(redshift, redshift))
#     # print(np.min(corr_samples))
#     plot_overview(corr_samples, actual_densities, temp_bright, "CORR", "./", redshift)


if True:
    for redshift in redshifts:
        # actual_densities = np.load("/Users/sabrinaberger/new_data_october/actual_densities_{}.npy".format(redshift))
        # temp_bright = np.load("/Users/sabrinaberger/new_data_october/temp_bright_{}.npy".format(redshift))
        # corr_samples = np.load("/Users/sabrinaberger/new_data_october/_num_samples_1000_eps_0_dvals_128_sigmaT_1_z_{}_CORRELATED_DENSITIES_{}_SAMPLES.npy".format(redshift, redshift))
        # #uncorr_samples = np.load("/Users/sabrinaberger/new_data_october/_num_samples_1000_eps_0_dvals_128_sigmaT_1_z_15_UNCORR_DENSITIES_15_SAMPLES.npy".format(redshift, redshift))
        actual_densities = np.load("{}_corr_densities_latest_run.npy".format(redshift))
        temp_bright = np.load("{}_corr_densities_latest_run.npy".format(redshift))
        uncorr_samples = np.load(direc + "/plots_num_samples_1000_eps_0_dvals_16_sigmaT_1_sigmaD_1_z_12_UNCORR_DENSITIES_12_SAMPLES.npy")

        neutral, ionized = get_neutral_ionized_boolean(temp_bright)

        if power:
            arr_const_pspec = []
            label_arr = [r"$P_{true}$", r"$P_{constructed}$", r"$P_{ionized=0}$"]
            colors = ["black", "purple", "red"]
            const_densities = corr_samples[-num_samples:]
            # const_densities = corr_samples
            for dens in const_densities:
                k_const, const_pspec = oned_power_spectrum(dens, 1, nbins)
                arr_const_pspec.append(const_pspec)
                # print("Finished a pspec.")
            screw_densities = np.copy(const_densities[-1]) # ionized pixels set to 0
            screw_densities[ionized] = 0
            k_screw, screw_pspec = oned_power_spectrum(screw_densities, 1, nbins)
            ks, true_pspec = oned_power_spectrum(actual_densities, 1, nbins)

            power_arr = [true_pspec, arr_const_pspec, screw_pspec]
            k_arr = [ks, k_const, k_screw]
            get_power_specs(k_arr, power_arr, label_arr, plot_direc, redshift, colors)

        if perc:
            errs_correlated = get_1sigma(corr_samples)
            errs_uncorrelated = get_1sigma(uncorr_samples)

            if any(neutral):
                errs_correlated_sigma_neutral = get_1sigma(np.transpose(corr_samples)[neutral])
            else:
                errs_correlated_sigma_neutral = 0

            if any(ionized):
                errs_correlated_sigma_ionized = get_1sigma(np.transpose(corr_samples)[ionized])
            else:
                errs_correlated_sigma_ionized = 0

            err_corr.append(errs_correlated)
            err_uncorr.append(errs_uncorrelated)

            err_corr_sigma_neutral.append(errs_correlated_sigma_neutral)
            err_corr_sigma_ionized.append(errs_correlated_sigma_ionized)

        if perc_68_over:
            typ = "corr"
            plot_overview_68perc(corr_samples, actual_densities, temp_bright, typ, plot_direc, redshift)

        if over:
            # samps = uncorr_samples
            # typ = "uncorr"
            # plot_overview(samps, actual_densities, temp_bright, typ, plot_direc, redshift, num_samples=1000)

            samps = uncorr_samples
            typ = "uncorr"
            plot_overview(samps, actual_densities, temp_bright, typ, plot_direc, redshift, num_samples=100)
            # size = len(actual_densities)
            # truths_data = actual_densities[-20:]
            # neutral = temp_bright > 0
            # ionized = temp_bright < 0
            # neut_arr = []
            # ioni_arr = []
            #
            # for i, neut in enumerate(neutral[39:45]):
            #     if neut:
            #         neut_arr.append(i + 39)
            #     else:
            #         ioni_arr.append(i + 39)
            #
            # xvals = np.linspace(0, size-1, size)
            #
            # fig_neutral, axes_neutral = triangle.plot_triangle(corr_samples, contours=True, color='y', colors='y', elements=neut_arr)
            # fig_ionized, axes_ionized = triangle.plot_triangle(corr_samples, contours=True, color='g', colors='g', fig=fig_neutral, axes=axes_ionized, elements=ioni_arr)
            # fig_ionized.savefig("{}/{}_ionized_neutral_corner_first5.pdf".format(plot_direc, redshift))

# np.save("err_corr.npy", err_corr)
# np.save("err_uncorr.npy", err_uncorr)
# np.save("err_corr_sigma_neutral.npy", err_corr_sigma_neutral)
# np.save("err_corr_sigma_ionized.npy", err_corr_sigma_ionized)

# err_corr = np.load("err_corr.npy")
# err_uncorr = np.load("err_uncorr.npy")
# err_corr_sigma_neutral = np.load("err_corr_sigma_neutral.npy")
# err_corr_sigma_ionized = np.load("err_corr_sigma_ionized.npy")

# plot_sigma_vs_z([err_corr, err_uncorr], ["correlated sigma", "uncorrelated sigma"], redshifts, plot_direc)
# plot_sigma_vs_z([err_corr_sigma_neutral, err_corr_sigma_ionized], ["neutral sigma", "ionized sigma"], redshifts, plot_direc)


# test = samples[-1000:, -20:]
# test_length = len(test)
# figure = corner.corner(test, show_titles=True, labels=xvals, truths=truths_data)
# figure.savefig("{}/{}_corner_last_{}.pdf".format(direc, type, redshift))
