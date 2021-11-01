import corner
import numpy as np
import matplotlib.pyplot as plt
from battaglia_full import Dens2bBatt
from battaglia_plus_bayes import DensErr

saving_direc = "/Users/sabrinaberger/new_data_october/"

eps = 0
size = 128

ones = np.ones(size)

cov_prior_uncorr = np.diag(ones)
cov_likelihood = np.diag(ones)

mu, sigma_T, sigma_D = 0, 1, 3
cov_likelihood *= sigma_T
cov_likelihood = cov_likelihood ** 2

# cov_prior = np.copy(cov_prior_uncorr)
# for i in range(np.shape(cov_prior)[0]):
#     if i != (np.shape(cov_prior)[0] - 1):
#         cov_prior[i][i+1] = 0.05
#         if i < (np.shape(cov_prior)[0] - 2):
#             cov_prior[i][i+2] = 0.05
#     if i != 0:
#         cov_prior[i][i-1] = 0.15
#         if i > 1:
#             cov_prior[i][i-2] = 0.05

cov_prior = np.copy(cov_prior_uncorr) * 10
for i in range(np.shape(cov_prior)[0]):
    if i != (np.shape(cov_prior)[0] - 1):
        cov_prior[i][i+1] = 1
        if i < (np.shape(cov_prior)[0] - 2):
            cov_prior[i][i+2] = 1
    if i != 0:
        cov_prior[i][i-1] = 3
        if i > 1:
            cov_prior[i][i-2] = 1


diags = np.diag(np.diag(cov_prior))
print("cov {}".format(cov_prior - np.diag(np.diag(cov_prior))))
# #correlated densities
x = np.random.randn(size)
L = np.linalg.cholesky(cov_prior)
correlated_vars = np.dot(L, x)
log_norm_correlated_density = np.exp(correlated_vars) - 1

print(log_norm_correlated_density)

redshifts = [15]
for z in redshifts:	
    dens2Tb = Dens2bBatt(log_norm_correlated_density, 1, z, one_d=True)
    temp_bright = dens2Tb.temp_brightness
    actual_densities = log_norm_correlated_density

    np.save(saving_direc + "temp_bright_{}.npy".format(z), temp_bright)
    np.save(saving_direc + "actual_densities_{}.npy".format(z), actual_densities)

    np.save(saving_direc + "cov_prior_{}.npy".format(z), cov_prior)
    #
    # actual_densities = np.load("actual_densities_{}.npy".format(z))
    # temp_bright = np.load("temp_bright_{}.npy".format(z))

    #correlated densities
    test1D = DensErr(z, actual_densities, temp_bright, cov_likelihood, cov_prior, sigma_D, sigma_T, epsilon=eps, actual_rhos=actual_densities, log_norm=True, nsamples=int(1000), emc=True, corner_plot=saving_direc, no_prior_uncorr=False)

    #uncorrelated densities
    #test1D = DensErr(z, actual_densities, temp_bright, cov_likelihood, cov_prior_uncorr, sigma_D, sigma_T, epsilon=eps, actual_rhos=actual_densities, log_norm=False, nsamples=int(1000), emc=True, corner_plot=saving_direc, no_prior_uncorr=True)
