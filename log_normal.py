from battaglia_plus_bayes import DensErr
import numpy as np
from sklearn import datasets

cov = datasets.make_spd_matrix(8, 8)
x = np.random.randn(8)
L = np.linalg.cholesky(cov)
correlated_density = np.dot(L, x)
log_norm_correlated_density = np.exp(correlated_density) - 1

temp_bright = 27*(1+log_norm_correlated_density) # TODO change to battaglia

eps = 0
z = 8
sigma_D = 10
test1D = DensErr(z, log_norm_correlated_density, temp_bright, cov, sigma_D, epsilon=eps, actual_rhos=log_norm_correlated_density, log_norm = True, nsamples=int(1e3), emc=True)