import corner
import numpy as np
import matplotlib.pyplot as plt
from battaglia_full import Dens2bBatt
from battaglia_plus_bayes import DensErr

eps = 0
size = 128

ones = np.ones(size)

cov_prior_uncorr = np.diag(ones)
cov_likelihood = np.diag(ones)

mu, sigma_T = 0, 1
cov_likelihood *= sigma_T
cov_likelihood = cov_likelihood ** 2

cov_prior = np.copy(cov_prior_uncorr) * 0.5
for i in range(np.shape(cov_prior)[0]):
    if i != (np.shape(cov_prior)[0] - 1):
        cov_prior[i][i+1] = 0.05
        if i < (np.shape(cov_prior)[0] - 2):
            cov_prior[i][i+2] = 0.05

    if i != 0:
        cov_prior[i][i-1] = 0.15
        if i > 1:
            cov_prior[i][i-2] = 0.05

np.save("cov_prior.npy", cov_prior)

sigma_D = 1
sigma_T = 1
#correlated densities
x = np.random.randn(size)
L = np.linalg.cholesky(cov_prior)
correlated_vars = np.dot(L, x)
log_norm_correlated_density = np.exp(correlated_vars) - 1
redshifts = [10, 8, 6, 4]
for z in redshifts:	
    dens2Tb = Dens2bBatt(log_norm_correlated_density, 1, z, one_d=True)
    temp_bright = dens2Tb.get_temp_brightness()

    np.save("actual_densities_{}.npy".format(z), log_norm_correlated_density)
    np.save("temp_bright_{}.npy".format(z), temp_bright)

    test1D = DensErr(z, log_norm_correlated_density, temp_bright, cov_likelihood, cov_prior, sigma_D, sigma_T, epsilon=eps, actual_rhos=log_norm_correlated_density, log_norm=True, nsamples=int(1e4), emc=True)

    #uncorrelated densities
    test1D = DensErr(z, log_norm_correlated_density, temp_bright, cov_likelihood, cov_prior_uncorr, sigma_D, sigma_T, epsilon=eps, actual_rhos=log_norm_correlated_density, log_norm=True, nsamples=int(1e4), emc=True)
