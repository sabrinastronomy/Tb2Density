import numpy as np
from battaglia_full import Dens2bBatt
# import corner
import glob
import h5py
from scipy import interpolate
import matplotlib.pyplot as plt
import statistics
import emcee
# import pyhmc

i = 0

class DensErr(Dens2bBatt):
    """
    This class applies an HMC to determine the errors on the density field predicted by the temperature brightness. It
    uses the Battaglia et al (2013) model in a Gaussian likelihood function.
    """

    def __init__(self, z, guess_density_field, obs_temp_brightness_field, cov_likelihood, cov_prior, sigma_prior, sigma_T, epsilon, actual_rhos, sigma_perturb=1, log_norm=True, one_d=True, delta_pos=1, delta_k=1, nsamples=1000, corner_plot="../STAT_DATA/plots", plotsamples=False, emc=False, pyMC3=False, pyHMC=False, no_prior_uncorr=False):
        self.z = z # setting the redshift
        self.Tarr = obs_temp_brightness_field # temperature brightness field we're trying to convert to density field
        self.prm_init = guess_density_field # initial guess for density field
        self.cov_likelihood = cov_likelihood
        self.delta_k = delta_k
        self.sigma_likelihood = np.sqrt(cov_likelihood[0][0])
        self.sigma_D = sigma_prior
        self.sigma_T = sigma_T
        self.currentBatt = "" # This is instance of Dens2bBatt including current density to temperature brightness calculation
        self.delta_pos = delta_pos # this varies if the density field is not at the same scale TODO
        self.nsamples = nsamples
        self.epsilon = epsilon
        self.plotsamples = plotsamples
        self.sigma_perturb = sigma_perturb
        self.log_norm = log_norm
        self.pyHMC = pyHMC
        self.pyMC3 = pyMC3
        self.no_prior_uncorr = no_prior_uncorr
        super().__init__(self.prm_init, self.delta_pos, self.z, one_d, False) # this is so we can access some parameters in Dens2bBatt that we need here

        if log_norm:
            # CORRELATED NAME
            self.cov_prior = cov_prior
            # checks if cov matrix is diagonal
            if np.count_nonzero(self.cov_prior - np.diag(np.diag(self.cov_prior))) != 0:
                self.corner_plot = corner_plot + "_num_samples_{}_eps_{}_dvals_{}_sigmaT_{}_z_{}_CORRELATED_DENSITIES".format(self.nsamples,
                                                                                                     self.epsilon,
                                                                                                     len(self.prm_init),
                                                                                                     sigma_T, self.z)
        elif self.no_prior_uncorr:
            # NO PRIOR DEBUG NAME
            self.corner_plot = corner_plot + "_num_samples_{}_eps_{}_dvals_{}_sigmaT_{}_sigmaD_{}_z_{}_NOPRIOR_DENSITIES".format(
                self.nsamples,
                self.epsilon,
                len(self.prm_init),
                self.sigma_T, self.sigma_D, z)
        else:
            # UNCORRELATED NAME
            self.corner_plot = corner_plot + "_num_samples_{}_eps_{}_dvals_{}_sigmaT_{}_sigmaD_{}_z_{}_UNCORR_DENSITIES".format(
                self.nsamples,
                self.epsilon,
                len(self.prm_init),
                self.sigma_T, self.sigma_D, z)


        # BE SURE TO DIFFERENTIATE BETWEEN THIS INSTANCE & INSTANCE OF CURRENT DENSITY BEING SAMPLED!
        self.emc = emc
        self.actual_rhos = actual_rhos
        if self.emc:
            self.run_emcee()
        elif self.pyHMC:
            from pyhmc import hmc
            self.run_pyHMC()

    def get_grad_prior(self, prm):
        # TODO: check prior grad
        prior_grad = -1/self.sigma_D**2 * prm
        return prior_grad

    def get_gradients(self, prm):
        """
        Returns
        -----------
        the gradients of the log posterior with respect to the current density field being sampled.
        """
        currentBatt = Dens2bBatt(prm, 1, self.z, one_d=True)
        # avoiding race conditions UGH!
        Tb = currentBatt.get_temp_brightness()
        X_HI = currentBatt.get_x_hi()
        z_re = currentBatt.get_z_re()
        set_z = currentBatt.set_z
        avg_z = currentBatt.avg_z

        final_grad = (self.grad_prefactor() * self.grad_sum(prm, X_HI, Tb, z_re, set_z, avg_z)) + self.get_grad_prior(prm)

        # print("final_grad {} for TMOD: {}, TMEAS: {}, RHO: {}, z_re {}, z {}, avg z {}, delta_z_loc {}".format(final_grad, self.Tarr, self.currentBatt.temp_brightness, self.currentBatt.density, self.currentBatt.z_re, self.currentBatt.set_z, self.currentBatt.avg_z, 0.1))
        return final_grad

    def grad_sum(self, prm, X_HI, Tb, z_re, set_z, avg_z):
        """
        Returns
        -----------
        returns final sum inside gradients
        """
        return X_HI * (self.Tarr - Tb) + self.long_sum(prm, z_re, set_z, Tb, avg_z, dz=0.1)

    def grad_prefactor(self):
        """
        Returns
        -----------
        gradient prefactor
        """
        return 27/(self.sigma_likelihood ** 2)

    def bias_interp(self):
        """
        Returns
        -----------
        Interpolated function for the bias factor which is the inverse FFT of r_a - r_j in either 3d or 1d
        """
        # inverse FFT
        if not self.one_d:
            # TODO 3D version
            kx_arr = np.arange(0, 300, 1)
            ky_arr = np.zeros(300)
            kz_arr = np.zeros(300)

            kx_arr *= 2 * np.pi  # scaling k modes correctly
            ky_arr *= 2 * np.pi  # scaling k modes correctly
            kz_arr *= 2 * np.pi  # scaling k modes correctly

            bias_factors = self.b_mz(kx_arr)
            bias_factors *= self.delta_k ** 3  # scaling amplitude in fourier space for 2D # TODO should this be the same?
            b_fourier = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(bias_factors, ky_arr, kz_arr)))
            b_fourier *= (len(kx_arr) ** 3) / (2 * np.pi) ** 3  # weird FFT scaling for 3D

            rx_arr = np.fft.ifftshift(np.fft.fftfreq(len(kx_arr), self.delta_pos)) # TODO should this be the same?
            # ry_arr = np.fft.ifftshift(np.fft.fftfreq(len(kx_arr), delta_pos))
            # rz_arr = np.fft.ifftshift(np.fft.fftfreq(len(kx_arr), delta_pos))

        elif self.one_d:
            # 1D version
            kx_arr = np.arange(0, 300, 1)
            kx_arr = (kx_arr * 2. * np.pi)  # scaling k modes correctly
            bias_factors = self.b_mz(kx_arr)
            bias_factors *= self.delta_k  # scaling amplitude in fourier space for 2D
            b_fourier = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(bias_factors)))
            b_fourier *= (len(kx_arr)) / (2 * np.pi)  # weird FFT scaling for 3D
            rx_arr = np.fft.ifftshift(np.fft.fftfreq(len(kx_arr), self.delta_pos))
            return interpolate.interp1d(rx_arr, b_fourier, fill_value="extrapolate")

        else:
            print("Dimensions of fields unavailable at this time!")
            raise TypeError



    def long_sum(self, prm, z_re, set_z, Tb, avg_z, dz=0.1):
        """
        Denominator in trig function.
        Parameters
        -----------
        densities: density field
        T_mod: model temperature field object

        Returns
        -----------
        gradient long sum
        gradient long sum
        """
        trig_input = (z_re - set_z)/dz
        trig_term = np.real((-2 * dz)**(-1) * (1/np.cosh(trig_input))**2)
        inside_sum_1 = self.Tarr - Tb
        inside_sum_2 = (1 + prm) * trig_term * (1 + avg_z)
        bias_function = self.bias_interp() # getting interpolated bias function

        densities = prm # current densities being sampled
        flattened = densities.flatten() # flattening density array
        if self.one_d:
            second_total_sum = np.empty(len(densities))
            for i, item in enumerate(flattened):
                bias_factors = []
                r_a = np.arange(len(flattened)) # we'll sum over this
                r_mags = np.sqrt((r_a - i) ** 2)
                bias_factors = bias_function(r_mags)  # TODO: add units
                second_total_sum[i] = np.sum(inside_sum_1*inside_sum_2*bias_factors)
        else:
            num_rows, num_cols = np.shape(densities)
            second_total_sum = np.empty((num_rows, num_cols))
            for i, item in enumerate(flattened):
                coord_i = get_row_col_depth(i, num_rows)
                bias_factors = []
                for a, item in enumerate(flattened):
                    coord_a = get_row_col_depth(a, num_cols)
                    r_mag = np.sqrt((coord_a[0]-coord_i[0])**2 + (coord_a[1]-coord_i[1])**2)
                    bias_factors.append(bias_function(r_mag, num_rows)) # TODO: add units
                second_total_sum[coord_i[0]][coord_i[1]] = np.sum(inside_sum_1*inside_sum_2*bias_factors)
        return second_total_sum

    @staticmethod
    def get_row_col_depth(x, num_rows):
        """
        Helper function

        Returns
        -----------
        coordinates within 3D array from a flattened array
        """
        return (np.floor(x/(2*num_rows)), x%num_rows, x)


    def numeric_posterior(self, prm):

        print("currently being tried: {}".format(prm))
        # LOGGED
        if np.min(prm) < -1:
            return -np.inf

        if self.one_d:
            currentBatt = Dens2bBatt(prm, 1, self.z, one_d=True)
            Tb = currentBatt.get_temp_brightness()

            if self.log_norm and (self.emc or self.pyMC3):
                # correlated prior
                c = np.linalg.det(self.cov_prior)
                log_norm_func = np.log(1 + prm)
                trans_prior = -(0.5) * log_norm_func.T
                trans_prior_C = np.dot(trans_prior, np.linalg.inv(self.cov_prior))
                trans_prior_C_prior = np.dot(trans_prior_C, log_norm_func)
                prefactor_prior = np.log(1 / (np.sqrt(2 * np.pi) * c))
                prior_rhos = trans_prior_C_prior

            else:
                if self.no_prior_uncorr:
                    prior_rhos = 0
                    prefactor_prior = 0
                else:
                    prior_rhos = -(0.5 / self.sigma_D ** 2) * np.sum((prm ** 2))
                    prefactor_prior = np.log(1 / (np.sqrt(2 * np.pi) * self.sigma_D) ** len(prm))

            #uncorrelated likelihood
            trans_X = (-0.5) * (Tb - self.Tarr).T
            trans_X_C = np.dot(trans_X, np.linalg.inv(self.cov_likelihood))
            trans_X_C_X = np.dot(trans_X_C, (Tb - self.Tarr))
            # print("trans_X_C_X {}".format(trans_X_C_X))

            c = np.linalg.det(self.cov_likelihood)
            # print("det cov likelihood {}".format(c))
            prefactor_likelihood = np.log(1 / (np.sqrt(2 * np.pi) * c) ** len(self.Tarr))
            # print("prefactor_likelihood {}".format(prefactor_likelihood))

            prefactor = (prefactor_likelihood + prefactor_prior)
            logp = prefactor + (prior_rhos + trans_X_C_X)
        # Keeping track of sample we're on
        global i
        i = i + 1
        if i % 1000:
           print("{} samples".format(i))

        if self.emc:
            return logp
        elif self.pyHMC:
            return logp, self.get_gradients(prm)
        elif self.pyMC3:
            print("logp: {}".format(logp))
            print("gradients: {}".format(self.get_gradients(prm)))
            return logp, self.get_gradients(prm)
        else:
            return logp


    def run_emcee(self):
        # 1) setting up walkers
        ndim, nwalkers = len(self.prm_init), len(self.prm_init)*2
        densities = np.copy(self.prm_init)
        x = []
        for n in range(nwalkers):
            # You could add a random seed density field
            perturb_rho = np.random.normal(0, 2, len(densities))
            # perturb_rho = np.random.normal(1, 2*self.sigma_perturb, len(densities))
            # adding = densities + perturb_rho
            # adding = 1 + perturb_rho
            for i, a in enumerate(perturb_rho):
                if a < -1:
                    perturb_rho[i] = -0.9
            print("perturbed densities {}".format(perturb_rho))
            # x.append(perturb_rho)
            x.append(perturb_rho)

        x = np.asarray(x)
        flattened_walkers = x.flatten()

        print("WALKERS: {}".format(flattened_walkers))
        print("RUNNING EMCEE")

        from datetime import datetime
        now = datetime.now()
        print("STARTED AT {}".format(now))

        # 2 Run the emcee below
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.numeric_posterior)
        sampler.run_mcmc(x, self.nsamples)
        samples = self.samples = sampler.get_chain(flat=True)
        # print("Final Samples: {}".format(samples))
        np.save('{}_{}_SAMPLES'.format(self.corner_plot, self.z), samples)

        modes = []
        for i, density in enumerate(np.transpose(samples)):
            modes.append(statistics.mode(density))

        if self.plotsamples:
            counts = np.linspace(1, len(self.prm_init), len(self.prm_init), dtype=int)
            figure = corner.corner(samples, show_titles=True, labels=counts, truths=self.actual_rhos)
            figure.savefig('{}_EMCEE.png'.format(self.corner_plot))
            plt.close(figure)
        now = datetime.now()
        print("FINISHED AT {}".format(now))


    def run_pyHMC(self):
        samples = hmc(self.numeric_posterior, x0=self.prm_init, n_samples=self.nsamples, epsilon=self.epsilon, return_diagnostics=True, return_logp=True, n_burn=1e3)
        self.rejection_rate = samples[2]["rej"]
        if self.plotsamples:
            counts = np.linspace(1, len(self.prm_init), len(self.prm_init), dtype=int)
            figure = corner.corner(samples[0], show_titles=True, labels=counts, truths=self.actual_rhos)
            figure.savefig('{}.png'.format(self.corner_plot))


if __name__ == "__main__":
    # starting with 1D
    z_get = 12  # it doesn't matter which z we pull the density field from
    z = 12
    one_d_size = 16

    ones = np.ones(one_d_size)
    cov = cov_prior = np.diag(ones)

    mu, sigma_T, sigma_D = 0, 1, 1
    cov *= sigma_T
    cov_likelihood = cov**2

    # squaring standard deviations to get to variance
    # noise = np.random.normal(mu, sigma_T, one_d_size)
    # Adding noise to temperature brightness
    # one_Tb += noise

    #### TESTING LIKELIHOOD
    actual_rho = np.asarray([1.92688, -0.41562])
    actual_Tb = np.asarray([79.02579, 15.77820])
    # test1D = DensErr(z, actual_rho, actual_Tb, cov_likelihood, cov_prior, sigma_D, sigma_T, epsilon=0, actual_rhos=actual_rho,
    #                  nsamples=int(500), emc=False, no_prior_uncorr=True, log_norm=False)
    # print("COMPARE {}".format(test1D.numeric_posterior(actual_rho)))


    cov_prior = np.copy(cov)
    for i in range(np.shape(cov_prior)[0]):
        if i != (np.shape(cov_prior)[0] - 1):
            cov_prior[i][i + 1] = 0.05
            if i < (np.shape(cov_prior)[0] - 2):
                cov_prior[i][i + 2] = 0.05
        if i != 0:
            cov_prior[i][i - 1] = 0.15
            if i > 1:
                cov_prior[i][i - 2] = 0.05

    # correlated densities
    # x = np.random.randn(one_d_size)
    # L = np.linalg.cholesky(cov_prior)
    # correlated_vars = np.dot(L, x)
    # one_rho = actual_rho = log_norm_correlated_density = np.exp(correlated_vars) - 1
    # dens2Tb = Dens2bBatt(one_rho, 1, z, one_d=True)
    # line_Tb = one_Tb = dens2Tb.temp_brightness

    ### Commented out if you want to use 21cmFAST generated field
    # dir = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/21cmFASTData/21cmFASTBoxes_{}/PerturbedField_*".format(z_get)
    # hf = h5py.File(glob.glob(dir)[0], 'r')
    # line_rho = hf["PerturbedField/density"][:, 0, 0]
    # dens2Tb = Dens2bBatt(line_rho, 1, z, one_d=True)
    # line_Tb = one_Tb = dens2Tb.temp_brightness
    # one_rho = actual_rho = line_rho[:one_d_size]
    # one_Tb = actual_Tb = line_Tb[:one_d_size]
    one_rho = np.load("{}_corr_densities_latest_run.npy".format(z))
    one_Tb = np.load("{}_corr_temp_brightness_latest_run.npy".format(z))

    # sigma_perturb = 0.1
    # perturb_rho = np.random.normal(0, sigma_perturb, one_d_size)
    # one_rho += perturb_rho

    print("sigma_D {}".format(sigma_D))
    # print("rho {}".format(one_rho))
    # print("TB {}".format(one_Tb))

    # actual_densities = one_rho = np.load("/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/new_32p_uncorr_actual_densities_15.npy")
    # one_Tb = np.load("/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/new_32p_uncorr_temp_bright_15.npy")
    # np.save("/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/new_32p_uncorr_actual_densities_15.npy", one_rho)
    # np.save("/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/new_32p_uncorr_temp_bright_15.npy", one_Tb)

    rej_rates = []
    i = 0
    epsilons = [1e-5]

    emc = True

    # if not emc:
    #     for eps in epsilons:
    #         i += 1
    #         test1D = DensErr(z, one_rho, one_Tb, cov_likelihood, cov_prior, sigma_D, sigma_T, epsilon=eps, nsamples=int(1e3), emc=False, log_norm=False)
    #         rej_rates.append(test1D.rejection_rate)

    # uncorrelated prior below works
    if emc:
        eps = 0
        test1D = DensErr(z, one_rho, one_Tb, cov_likelihood, cov_prior, sigma_D, sigma_T, epsilon=eps, actual_rhos=one_rho, nsamples=int(1e4), emc=True, no_prior_uncorr=False, log_norm=False)

    # if emc:
        # eps = 0
        # test1D = DensErr(z, one_rho, one_Tb, cov_likelihood, cov_prior, sigma_D, sigma_T, epsilon=eps, actual_rhos=one_rho, nsamples=int(500), emc=True, no_prior_uncorr=False, log_norm=False)

    # transposed_samps = np.load("../STAT_DATA/plots_num_samples_500_eps_0_dvals_32_sigmaT_1_sigmaD_1_z_15_UNCORR_DENSITIES_15_SAMPLES.npy")[-1]

    # GET log likelihood of last samples and actual densities
    # print("log likelihood actual {}".format(test1D.numeric_posterior(actual_densities)))
    # print("log likelihood samples {}".format(test1D.numeric_posterior(test1D.samples[-1])))


    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(one_d_size)
    # fig.suptitle('{} Trace Plots'.format("DEBUG"))
    #
    # for i, d in enumerate(transposed_samps[:5]):
    #     print(actual_rho)
    #     axs[i].plot(d, color="purple")
    #     axs[i].hlines(actual_rho[i], xmin=0, xmax=len(d), color="green", zorder=100)
    #
    # plt.savefig("testing.png", dpi=200)


