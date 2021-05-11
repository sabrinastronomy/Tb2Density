import corner
import numpy as np
import matplotlib.pyplot as plt
from battaglia_full import Dens2bBatt
from battaglia_plus_bayes import DensErr

z=8
size = 8

ones = np.ones(size)

cov_prior_uncorr = np.diag(ones)
cov_likelihood = np.diag(ones)

mu, sigma_T = 0, 1
cov_likelihood *= sigma_T
cov_likelihood = cov_likelihood ** 2


# TO DO: make Gaussian that peaks at 1
cov_prior = np.copy(cov_prior_uncorr) * 0.5
for i in range(np.shape(cov_prior)[0]):
    if i != (np.shape(cov_prior)[0] - 1):
        cov_prior[i][i+1] = 0.15
        if i < (np.shape(cov_prior)[0] - 2):
            cov_prior[i][i+2] = 0.05

    if i != 0:
        cov_prior[i][i-1] = 0.15
        if i > 1:
            cov_prior[i][i-2] = 0.05


# cov_prior[1, 0] = 0
# cov_prior[1, 1] = 1.2
# cov_prior[2, 1] = 1.2
# cov_prior[3, 2] = 1


print(cov_prior)

eps = 0
z = 8
sigma_D = 1
sigma_T = 1
samples = []
#correlated densities
# for i in range(int(1e3)):
x = np.random.randn(size)
L = np.linalg.cholesky(cov_prior)
correlated_vars = np.dot(L, x)
log_norm_correlated_density = np.exp(correlated_vars) - 1
dens2Tb = Dens2bBatt(log_norm_correlated_density, 1, z, one_d=True)
samples.append(log_norm_correlated_density)
temp_bright = dens2Tb.get_temp_brightness()

plt.plot(log_norm_correlated_density, marker="o", label="Correlated Density")
plt.plot(temp_bright, marker="o", label="Temperature Brightness")
plt.legend()
plt.savefig("density.pdf")
print('saved density plot')

# print("input prms {}".format(log_norm_correlated_density))
# print("temp brightness {}".format(temp_bright))

test1D = DensErr(z, log_norm_correlated_density, temp_bright, cov_likelihood, cov_prior, sigma_D, sigma_T, epsilon=eps, actual_rhos=log_norm_correlated_density, log_norm=True, nsamples=int(1e3), emc=True)

#uncorrelated densities
test1D = DensErr(z, log_norm_correlated_density, temp_bright, cov_likelihood, cov_prior_uncorr, sigma_D, sigma_T, epsilon=eps, actual_rhos=log_norm_correlated_density, log_norm=True, nsamples=int(1e3), emc=True)





