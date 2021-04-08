import numpy as np
from pyhmc import hmc
import corner

# def logprob(x, ivar):
#     logp = -0.5 * np.sum(ivar * x**2)
#     grad = -ivar * x
#     return logp, grad

def gradients(prm, alpha, cov, Tarr):
    # placeholder
    densities = prm # what we're sampling
    sigma_2 = cov[0][0] # sigma^2, one element of cov matrix
    grads = (1/sigma_2**2) * (-alpha*densities**2 + Tarr) # calculated gradients for each density value
    return grads

def numeric_posterior(m, x, b):

    return num_post, gradients(prm, alpha, cov, Tarr)

samples = hmc(numeric_posterior, x0=[.5, 1.5, 1], args=(alpha, cov, Tarr), n_samples=int(1e6))
# print(samples)
#   # pip install triangle_plot
figure = corner.corner(samples)
figure.savefig('testing_more_samples.png')
