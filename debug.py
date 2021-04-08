import glob

import corner
import h5py
import numpy as np
from pyhmc import hmc

from battaglia_full import Dens2bBatt

z = 12
one_d_size = 4
dir = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/building/21cmFASTBoxes_12/PerturbedField_*"
hf = h5py.File(glob.glob(dir)[0], 'r')
density_field_from_21cmFAST = hf["PerturbedField/density"]
density3D = np.asarray(density_field_from_21cmFAST)
dens2Tb = Dens2bBatt(density3D, 1, z, one_d=False)
Tb = dens2Tb.temp_brightness
line_rho = density3D[:, 0, 0]
line_Tb = Tarr = Tb[:, 0, 0]

# Adding noise to temperature brightness
ones = np.ones(one_d_size)
one_Tb = line_Tb[:one_d_size]
print("actual temperature values {}".format(one_Tb))

mu, sigma_T = 0, 0.1
noise = np.random.normal(mu, sigma_T, one_d_size)
one_Tb += noise
print("perturbed temperature values {}".format(one_Tb))

# Adding perturbation to density field
sigma_D = sigma_T / 27
one_rho = line_rho[:one_d_size]
print(sigma_D)
print("actual density values {}".format(one_rho))

perturb_rho = np.random.normal(0, 0.1 * sigma_D, one_d_size)
one_rho += perturb_rho
T_obs = one_Tb
print("perturbed density values {}".format(one_rho))

cov = np.diag(ones)
cov *= sigma_T
cov = cov ** 2

def logprob(density):
    # temperature brightness gaussian likelihood works
    # testing prior
    prior_rhos = -(0.5 / sigma_D ** 2) * np.sum((density ** 2))
    prefactor_prior = 1 / (np.sqrt(2 * np.pi) * sigma_D) ** len(density)

    trans_X = (-0.5) * (27 * density + 27 - T_obs).T
    trans_X_C = np.dot(trans_X, np.linalg.inv(cov))
    trans_X_C_X = np.dot(trans_X_C, (27 * density + 27 - T_obs))

    # logp = -(0.5 / sigma_T ** 2) * np.sum((27 * density + 27 - T_obs) ** 2)
    grad = - ((27 * density + 27 - T_obs) * 27) / sigma_T ** 2

    c = np.linalg.det(cov)
    prefactor_likelihood = 1 / (np.sqrt(2 * np.pi) * c) ** len(T_obs)
    prefactor = (prefactor_likelihood + prefactor_prior)
    logp = prefactor + (prior_rhos + trans_X_C_X)
    return logp, grad

def withCov_logprob(density):
    c = np.linalg.det(cov)
    prefactor_likelihood = np.log(1 / (np.sqrt(2 * np.pi) * c) ** len(T_obs))
    trans_X = (-0.5) * (27 * density + 27 - T_obs).T
    trans_X_C = np.dot(trans_X, np.linalg.inv(cov))
    trans_X_C_X = np.dot(trans_X_C, (27 * density + 27 - T_obs))
    logp = trans_X_C_X * prefactor_likelihood
    grad = - ((27 * density + 27 - T_obs) * 27) / sigma_T ** 2
    return logp, grad

og = logprob(one_rho)
og_new = withCov_logprob(one_rho)
for eps in [0.0001]:
# for eps in [0.0001]:
    samples = hmc(logprob, x0=one_rho, n_samples=int(1e5), epsilon=eps)
    figure = corner.corner(samples, show_titles=True, labels=["d1", "d2", "d3", "d4"])
    figure.savefig('../STAT_IMAGES/all_pngs/testing_{}.png'.format(eps))
# samples *= sigma_D
