import numpy as np
from pyhmc import hmc
import corner

i = 0

def gradients(prm, cov, Tarr):
    # placeholder
    alpha = prm[0]
    densities = prm[1:len(prm)] # what we're sampling
    sigma_2 = cov[0][0] # sigma^2, one element of cov matrix
    alpha_grad_sum = -np.sum(densities**2) + np.sum(Tarr * densities**2)
    alpha_grad = [(sigma_2) * alpha_grad_sum]
    grads_dens = (sigma_2) * (Tarr * 2 * alpha * densities - 2 * alpha**2 * densities**3) # calculated gradients for each density value
    return np.concatenate((alpha_grad, grads_dens))

def numeric_posterior(prm, cov, Tarr, PRIOR_sigma):
    # LOGGED

    alpha = prm[0]
    rho_vec = np.vstack(np.asarray(prm[1:len(prm)]))
    sigma_a = PRIOR_sigma[0]
    sigma_d = PRIOR_sigma[1]
    T_vec = np.vstack(Tarr)
    prior_rhos = np.sum((-rho_vec ** 2 / (2 * sigma_d ** 2)))
    prior_alpha = (-alpha ** 2 / (2 * sigma_a ** 2))
    c = np.linalg.det(cov)
    exp_1 = (-0.5) * (T_vec - (alpha * rho_vec ** 2)).T
    exp_1 = np.dot(exp_1, np.linalg.inv(cov))
    exp_1 = np.dot(exp_1, (T_vec - (alpha * rho_vec ** 2))).flatten()
    prefactor_prior = np.log(1 / (2 * np.pi * (sigma_a * sigma_d ** (len(rho_vec)))))
    prefactor_likelihood = np.log(1 / (2 * np.pi * c))
    num_post = prefactor_likelihood * prefactor_prior * (exp_1 + prior_alpha + prior_rhos)
    global i
    i = i + 1
    if i % 1000:
        print("{} samples".format(i))
    return num_post, gradients(prm, cov, Tarr)

cov = np.asarray([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
Tarr = np.asarray([1, 1, 1])
PRIOR_sigma = [2, 2]
# print(numeric_posterior(np.asarray([1, 1, 1]), alpha, cov, Tarr)
samples = hmc(numeric_posterior, x0=[1, 1, 1, 1], args=(cov, Tarr, PRIOR_sigma), n_samples=int(1e6))

figure = corner.corner(samples, labels=["alpha", "d_1", "d_2", "d_3"], show_titles=True, title_kwargs={"fontsize": 12})
figure.savefig('adrian_sigma_2_3.png')
