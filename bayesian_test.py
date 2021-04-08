import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from sklearn import datasets
import pymc3 as pm
import os


class Bayes:
    def __init__(self, type, dimensions, posterior, temperatures, sigmas, deltas=np.linspace(0, 50, 100), covariance_matrix=[], numeric=False, go=True, hmc=True, direc=os.getcwd()):
        self.name = type
        self.dimensions = dimensions
        self.posterior = posterior
        self.temperatures = temperatures
        self.sigmas = sigmas
        self.deltas = deltas
        self.numeric = numeric
        self.probs = []
        self.cov = covariance_matrix
        self.hmc = hmc
        self.direc = direc
        # Ensure that covariance matrix is positive definite
        if len(self.cov) > 0 and not np.all(np.linalg.eigvals(self.cov) > 0):
            raise ValueError("Covariance matrix is not positive definite!")
        if go:
            self.go()


    def find_posteriors(self):
        if self.numeric:
            nwalkers = 32
            if not self.hmc:
                p0 = np.random.uniform(0, 1, size=(nwalkers, self.dimensions))  # guessing random number between 0 and 1 for each element
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.posterior,
                                                args=[self.temperatures, *self.sigmas, self.cov])
                sampler.run_mcmc(p0, 50000)
                self.probs = sampler.get_chain(flat=True)
            else:
                with pm.Model():
                    sigma_1, sigma_2, sigma_a, sigma_d = self.sigmas[0], self.sigmas[1], self.sigmas[2], self.sigmas[3]
                    pm.DensityDist('likelihood', self.posterior,
                                   observed={'prm': self.deltas, 'sigma_1': 1, 'sigma_2': 2, 'Tarr': self.temperatures, 'cov': self.cov, "sigma_a": sigma_a, "sigma_d": sigma_d})

                    step = pm.NUTS()
                    trace = pm.sample(2000, tune=1000, init=None, step=step)
        else: # Finding posterior analytically
            self.probs = self.posterior(*self.temperatures, *self.sigmas)(self.deltas)
            self.normalize_posterior()

    def normalize_posterior(self):
        return self.probs / (np.sum(self.probs) * (self.deltas[1] - self.deltas[0]))

    def plotter(self):
        if not self.numeric:
            if self.name == "one_pix":
                plt.ylabel("$P(\delta | T)$")
                plt.semilogy(self.deltas, self.probs, label="T={} mK".format(*self.temperatures))
            if self.name == "two_pix_uncorr_no_noise":
                plt.ylabel("$P(\delta | T1, T2)$")
                plt.semilogy(self.deltas, self.probs, label="T={} & {} mK".format(*self.temperatures))
            if self.name == "two_pix_uncorr_w_noise":
                plt.ylabel("$P(\delta | T1, T2)$")
                plt.semilogy(self.deltas, self.probs, label="T={} & {} mK".format(*self.temperatures))
            plt.title(r"Toy Inference Problem ({}): $T=\alpha \delta^2$".format(self.name))
            plt.xlabel("$\delta$")
            plt.legend()
            plt.savefig(self.direc + "{}.png".format(self.name))
            plt.show()
        else:
            for n in range(np.shape(self.probs)[1]):
                plt.hist(self.probs[:, n], 100, color="k", histtype="step", density=True)
                plt.xlabel(r"{} param".format(n))
                plt.ylabel(r"$p({} param)$".format(n))
                plt.savefig(self.direc + "{}_param_test.pdf".format(n))
                plt.close()

            fig, axes = plt.subplots(1, ndim)
            fig.suptitle(r'$P(\alpha, \delta_1, \delta_2 | T_1, T_2 $', y=1.05)
            count = 0
            for ax in axes:
                ax.hist(self.probs[:, count], 100, color="k", histtype="step", density=True)
                plt.xlabel(r"{} param".format(n))
                plt.ylabel(r"$p({} param)$".format(n))
                count += 1
            fig.savefig(self.direc + "1d_{}.pdf".format(self.name))
            # labels = np.full("param_")
            counts = np.linspace(1, ndim, ndim, dtype=int)
            # labels = [x + count for x, count in zip(labels, counts)]
            figure = corner.corner(self.probs, labels=counts,
                                   show_titles=True, title_kwargs={"fontsize": 12})
            figure.savefig(self.direc + "corner_test_{}.pdf".format(self.name))

    def go(self):
        self.find_posteriors()
        self.plotter()

# ANALYTIC posteriors for toy model: T=alpha delta^2
def onepix_marginalized_posterior(T, sigma_alpha, sigma_delta):
    return lambda delta: (1 / delta ** 2) * np.exp(-delta ** 2 / (2 * (sigma_delta ** 2))) / (
                2 * np.pi * sigma_alpha * sigma_delta) * np.exp((-T ** 2 / delta ** 4) / (2 * (sigma_alpha ** 2)))

def twopix_marginalized_posterior(T_1, T_2, sigma_alpha, sigma_delta, noise=False):
    prefactor = lambda delta: (delta * np.sqrt(T_1)) / (
                4 * np.sqrt(T_1) * T_1 * np.pi * sigma_alpha * (sigma_delta ** 2))
    exp_1 = lambda delta: np.exp(-delta ** 2 / (2 * (sigma_delta ** 2)))
    exp_2 = lambda delta: np.exp((-T_2 ** 2 * delta ** 2) / (2 * T_1 * sigma_delta ** 2))
    exp_3 = lambda delta: np.exp((-T_1 ** 2) / (2 * delta ** 4 * sigma_alpha))
    if noise:
        return
    else:
        return lambda delta: prefactor(delta) * exp_1(delta) * exp_2(delta) * exp_3(delta)

T_1 = 10
T_2 = 11
sigma_alpha = 10
sigma_delta = 11

# 2.1: 1 pixel, no noise (Dirac Delta likelihood)
onepix_marginalized_uncorr = Bayes("one_pix", 1, onepix_marginalized_posterior, [T_1], [1, 2])
# 2.2: 2 pixels uncorrelated, no noise (Dirac Delta likelihood)
twopix_marginalized_uncorr = Bayes("two_pix_uncorr_no_noise", 1, twopix_marginalized_posterior, [T_1, T_2], [sigma_alpha, sigma_delta])


# NUMERICALLY calculating posteriors for toy model: T=alpha delta^2
def numeric_posterior(prm, Tarr, sigma_1, sigma_2, sigma_a, sigma_d, cov=[]):
    # LOGGED
    # alpha: prm[0]
    # densities: prm[1:2]
    if len(cov) > 0:
        c = np.linalg.det(cov)
        rho_vec = np.vstack(np.asarray(prm[1:]))
        T_vec = np.vstack(Tarr)
        alpha = prm[0]

        exp_1 = (-0.5) * (T_vec - (alpha * rho_vec ** 2)).T
        exp_1 = np.matmul(exp_1, np.linalg.inv(cov))
        exp_1 = np.matmul(exp_1, (T_vec - (alpha * rho_vec ** 2)))
        prior = (-rho_vec ** 2 / (2 * sigma_d ** 2)) + (-alpha ** 2 / (2 * sigma_a ** 2))
        prefactor_prior = np.log(1 / (2 * np.pi * (sigma_a * sigma_d ** (len(rho_vec)))))
        prefactor_likelihood = np.log(1 / (2 * np.pi * c))
        return prefactor_likelihood * prefactor_prior * (prior + exp_1)
    else:
        # two pixel
        T_1 = Tarr[0]
        T_2 = Tarr[1]
        exp_1 = (-0.5) * ((T_1 - (prm[0] * prm[1] ** 2)) ** 2) / (2 * sigma_1 ** 2)
        exp_2 = (-0.5) * ((T_2 - (prm[0] * prm[2] ** 2)) ** 2) / (2 * sigma_2 ** 2)
        prior = (-prm[1] ** 2 / (2 * sigma_d ** 2)) + (-prm[2] ** 2 / (2 * sigma_d ** 2)) + (
                    -prm[0] ** 2 / (2 * sigma_a ** 2))
        prefactor_likelihood = np.log(1 / (2 * np.pi * (sigma_1 * sigma_2) ** 2))
        prefactor_prior = np.log(1 / (2 * np.pi * (sigma_a * sigma_d ** 2)))
        return prefactor_likelihood * prefactor_prior * (exp_1 + exp_2 + prior)

ndim = 5
sigma_1 = 5
sigma_2 = 5
x1x2_ens = 4
sigma_a = 2
sigma_d = 3
cov_uncorr = np.array([[sigma_1, 0], [0, sigma_2]])
cov_corr_two = np.array([[sigma_1, x1x2_ens], [x1x2_ens, sigma_2]])  # ensure positive definite: symmetric and eigenvalues are all > 0
cov = np.abs(datasets.make_spd_matrix(ndim-1)*10)
print(cov)

# twopix_gaussian_uncorr = Bayes("two_pix_uncorr_noise_gauss", ndim, numeric_posterior, [T_1, T_2], [sigma_1, sigma_2, sigma_a, sigma_d], covariance_matrix=cov_uncorr, numeric=True)
# twopix_gaussian_corr = Bayes("two_pix_corr_noise_gauss", ndim, numeric_posterior, [T_1, T_2], [sigma_1, sigma_2, sigma_a, sigma_d], covariance_matrix=cov_corr_two, numeric=True)


ndim = 11
cov = np.abs(datasets.make_spd_matrix(ndim-1)*10)
tenpix_gaussian_corr = Bayes("five_pix_corr_noise_gauss", ndim, numeric_posterior, np.linspace(10, 100, ndim-1), [sigma_1, sigma_2, sigma_a, sigma_d], covariance_matrix=cov, numeric=True)


# tenpix_gaussian_corr = Bayes("ten_pix_corr_noise_gauss", ndim, numeric_posterior, np.linspace(10, 100, ndim-1), [sigma_1, sigma_2, sigma_a, sigma_d], covariance_matrix=cov, numeric=True)

