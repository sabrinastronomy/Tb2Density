import numpy as np
from pyhmc import hmc
from battaglia_full import Dens2bBatt
import corner
import glob
import h5py
from scipy import interpolate
import matplotlib.pyplot as plt
import emcee
import statistics

i = 0

class DensErr(Dens2bBatt):
    """
    This class applies an HMC to determine the errors on the density field predicted by the temperature brightness. It
    uses the Battaglia et al (2013) model in a Gaussian likelihood function.
    """

    def __init__(self, z, guess_density_field, temp_brightness_field, cov_likelihood, cov_prior, sigma_prior, sigma_T, epsilon, actual_rhos, sigma_perturb=0.1, log_norm=False, one_d=True, delta_pos=1, delta_k=1, nsamples=1000, corner_plot="../plots", plotsamples=True, emc=True):
        self.z = z # setting the
        self.Tarr = temp_brightness_field # temperature brightness field we're trying to convert to density field
        self.prm_init = guess_density_field # initial guess for density field
        self.prm_curr = ""
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
        super().__init__(self.prm_init, self.delta_pos, self.z, one_d, False) # this is so we can access some parameters in Dens2bBatt that we need here

        if log_norm:
            self.cov_prior = cov_prior
            # checks if cov matrix is diagonal
            if np.count_nonzero(self.cov_prior - np.diag(np.diagonal(self.cov_prior))) != 0:

                self.corner_plot = corner_plot + "_num_samples_{}_eps_{}_dvals_{}_sigmaT_{}_z_{}_CORRELATED_DENSITIES".format(self.nsamples,
                                                                                                     self.epsilon,
                                                                                                     len(self.prm_init),
                                                                                                     sigma_T, self.z)
            else:
                self.corner_plot = corner_plot + "_num_samples_{}_eps_{}_dvals_{}_sigmaT_{}_z_{}_UNCORR_DENSITIES".format(
                    self.nsamples,
                    self.epsilon,
                    len(self.prm_init),
                    sigma_T, self.z)

        else:
            self.corner_plot = corner_plot + "_num_samples_{}_eps_{}_dvals_{}_sigmaT_{}_z_{}".format(self.nsamples, self.epsilon, len(self.prm_init), sigma_T, self.z)
        # BE SURE TO DIFFERENTIATE BETWEEN THIS INSTANCE & INSTANCE OF CURRENT DENSITY BEING SAMPLED!
        self.emc = emc
        self.actual_rhos = actual_rhos
        if self.emc:
            self.run_emcee()
        else:
            self.run_HMC()

    def get_grad_prior(self):
        # TODO: check prior grad
        self.prior_grad = -1/self.sigma_D**2 * self.prm_curr
        return self.prior_grad

    def get_gradients(self):
        """
        Returns
        -----------
        the gradients of the log posterior with respect to the current density field being sampled.
        """
        final_grad = (self.grad_prefactor() * self.grad_sum()) + self.get_grad_prior()
        # print("final_grad {} for TMOD: {}, TMEAS: {}, RHO: {}, z_re {}, z {}, avg z {}, delta_z_loc {}".format(final_grad, self.Tarr, self.currentBatt.temp_brightness, self.currentBatt.density, self.currentBatt.z_re, self.currentBatt.set_z, self.currentBatt.avg_z, 0.1))
        return final_grad

    def grad_sum(self):
        """
        Returns
        -----------
        returns final sum inside gradients
        """
        return self.currentBatt.X_HI * (self.Tarr - self.currentBatt.temp_brightness) + self.long_sum()

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



    def long_sum(self, dz=0.1):
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
        trig_input = (self.currentBatt.z_re - self.currentBatt.set_z)/dz
        trig_term = np.real((-2 * dz)**(-1) * (1/np.cosh(trig_input))**2)
        inside_sum_1 = self.Tarr - self.currentBatt.get_temp_brightness()
        inside_sum_2 = (1 + self.prm_curr) * trig_term * (1 + self.currentBatt.avg_z)
        bias_function = self.bias_interp() # getting interpolated bias function

        densities = self.prm_curr # current densities being sampled
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
        # print("second_total_sum {}".format(second_total_sum))
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
        # LOGGED
        self.prm_curr = prm
        if np.min(self.prm_curr) < -1:
            return -np.inf

        if self.one_d:
            self.currentBatt = Dens2bBatt(prm, 1, self.z, one_d=True)
            Tb = self.currentBatt.get_temp_brightness()
            if self.log_norm and self.emc:
                #correlated prior
                c = np.linalg.det(self.cov_prior)
                # print("input prms {}".format(self.prm_curr))
                log_norm_func = np.log(1 + self.prm_curr)
                # print("log_norm_func {}".format(log_norm_func))
                trans_prior = -(0.5) * log_norm_func.T
                trans_prior_C = np.dot(trans_prior, np.linalg.inv(self.cov_prior))
                trans_prior_C_prior = np.dot(trans_prior_C, log_norm_func)
                # print("prior_rhos {}".format(trans_prior_C_prior))
                prefactor_prior = 1 / (np.sqrt(2 * np.pi) * c)
                prior_rhos = trans_prior_C_prior
                # print("prefactor_prior {}".format(prefactor_prior))

            else:
                prior_rhos = -(0.5 / self.sigma_D ** 2) * np.sum((prm ** 2))
                prefactor_prior = 1 / (np.sqrt(2 * np.pi) * self.sigma_D) ** len(prm)

            # uncorrelated likelihood
            trans_X = (-0.5) * (Tb - self.Tarr).T
            trans_X_C = np.dot(trans_X, np.linalg.inv(self.cov_likelihood))
            trans_X_C_X = np.dot(trans_X_C, (Tb - self.Tarr))

            c = np.linalg.det(self.cov_likelihood)
            prefactor_likelihood = 1 / (np.sqrt(2 * np.pi) * c) ** len(self.Tarr)

            # prefactor = (prefactor_likelihood)
            # logp = prefactor + (trans_X_C_X)

            prefactor = (prefactor_likelihood + prefactor_prior)
            # print("prefactor {}".format(prefactor))

            logp = prefactor + (prior_rhos + trans_X_C_X)
            # print("logp {}".format(logp))

        # Keeping track of sample we're on
        # global i
        # i = i + 1
        # if i % 1000:
        #     # print("{} samples".format(i))
        # print("num_post {}".format(num_post)) # NOT normalized posterior
        if self.emc:
            return logp
        else:
            return logp, self.get_gradients()

    def run_emcee(self):
        ndim, nwalkers = len(self.prm_init), 3*len(self.prm_init)
        # p0 = np.random.randn(nwalkers, ndim)
        densities = np.copy(self.prm_init)
        x = np.copy(self.prm_init)
        x = [x]
        for n in range(nwalkers-1):
            perturb_rho = np.random.normal(0, 5*self.sigma_perturb, len(densities))
            adding = densities+perturb_rho
            x.append(adding)
        print("RUNNING EMCEE")
        import time
        print("STARTED AT {}".format(time.time()))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.numeric_posterior)
        sampler.run_mcmc(x, self.nsamples)

        samples = sampler.get_chain(flat=True)
        np.save('{}_SAMPLES'.format(self.corner_plot), samples)
        i = 0
        modes = []
        for i, density in enumerate(np.transpose(samples)):
            modes.append(statistics.mode(density))

        print("modes {}".format(modes))

        if self.plotsamples:
            counts = np.linspace(1, len(self.prm_init), len(self.prm_init), dtype=int)
            figure = corner.corner(samples, show_titles=True, labels=counts, truths=self.actual_rhos)
            figure.savefig('{}_EMCEE.png'.format(self.corner_plot))
            plt.close(figure)

    def run_HMC(self):

        samples = hmc(self.numeric_posterior, x0=self.prm_init, n_samples=self.nsamples, epsilon=self.epsilon, return_diagnostics=True, return_logp=True, n_burn=1e3)
        print(samples)
        self.rejection_rate = samples[2]["rej"]
        if self.plotsamples:
            counts = np.linspace(1, len(self.prm_init), len(self.prm_init), dtype=int)
            figure = corner.corner(samples[0], show_titles=True, labels=counts, truths=self.actual_rhos)
            figure.savefig('{}.png'.format(self.corner_plot))


if __name__ == "__main__":
    # starting with 2D
    z = 8
    # one_d_size = 16
    dir = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/21cmFASTData/21cmFASTBoxes_{}/PerturbedField_*".format(z)
    hf = h5py.File(glob.glob(dir)[0], 'r')
    line_rho = hf["PerturbedField/density"][:, 0, 0]
    dens2Tb = Dens2bBatt(line_rho, 1, z, one_d=True)
    line_Tb = dens2Tb.temp_brightness
    # print("TB {}".format(Tb))


    one_Tb = actual_Tb = line_Tb[41:57]
    one_rho = actual_rho = line_rho[41:57]
    one_d_size = len(one_Tb)
    # Adding noise to temperature brightness
    ones = np.ones(one_d_size)
    cov = np.diag(ones)


    print("rho {}".format(one_rho))
    print("TB {}".format(one_Tb))

    mu, sigma_T = 0, 1
    cov *= sigma_T
    cov_likelihood = cov**2 # squaring standard deviations to get to variance
    noise = np.random.normal(mu, sigma_T, one_d_size)
    one_Tb += noise

    # adding one ionized pixel!
    # one_Tb[0] = 0
    # Adding perturbation to density field
    sigma_D = sigma_T/27 * 1e3
    sigma_perturb = 0.1
    perturb_rho = np.random.normal(0, sigma_perturb, one_d_size)
    one_rho += perturb_rho
    print("sigma_D {}".format(sigma_D))
    print("rho {}".format(one_rho))
    # print("TB {}".format(one_Tb))

    rej_rates = []
    i = 0
    epsilons = [1e-5]

    emc = True

    if not emc:
        for eps in epsilons:
            i += 1
            test1D = DensErr(z, one_rho, one_Tb, cov_likelihood, cov_prior, sigma_D, sigma_T, epsilon=eps, nsamples=int(1e4), emc=False)
            rej_rates.append(test1D.rejection_rate)

    if emc:
        eps = 0
        test1D = DensErr(z, one_rho, one_Tb, cov_likelihood, cov_prior, sigma_D, sigma_T, epsilon=eps, actual_rhos=actual_rho, nsamples=int(1e4), emc=True)

    # import matplotlib.pyplot as plt
    # plt.close()
    # plt.clf()
    # plt.semilogx(epsilons, rej_rates)
    # plt.xlabel("Epsilons")
    # plt.ylabel("Acceptance Rates")
    # plt.savefig("esp_log_scale_{-6, -4}.pdf")


