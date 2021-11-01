import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
import cython
import matplotlib.pyplot as plt
import warnings
from battaglia_plus_bayes import DensErr
from battaglia_full import Dens2bBatt

#-----------------------------------------------------------------------------

z = 8
sigma_D = 1
sigma_T = 1
eps = 0
ndim = size = 2

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

temp_bright = np.load("/Users/sabrinaberger/new_data/temp_bright_8.npy")[70:72]
actual_densities = np.load("/Users/sabrinaberger/new_data/actual_densities.npy")[70:72]
print("actual_densityies {}".format(actual_densities))
bayes_model = DensErr(z, actual_densities, temp_bright, cov_likelihood, cov_prior, sigma_D, sigma_T, pyMC3=True, epsilon=eps, actual_rhos=actual_densities, nsamples=0, log_norm=False, emc=False)


ndraws = 1000  # number of draws from the distribution
nburn = 10   # number of "burn-in points" (which we'll discard)
#-----------------------------------------------------------------------------

# define your really-complicated likelihood function that uses loads of external codes
def my_model(prm):
    currentBatt = Dens2bBatt(prm, 1, z, one_d=True)
    Tb4Dens = currentBatt.get_temp_brightness()
    return Tb4Dens

def my_model_random(point=None, size=None):
    """
    Draw posterior predictive samples from model.
    """
    points = ()
    for p in point:
        points.append(p)

    return my_model(points)

def my_loglike(theta):
    """
    A Gaussian log-likelihood function for a model with parameters given in theta
    """
    # print(theta)
    posterior = bayes_model.numeric_posterior(theta)
    # print("posterior {}".format(posterior))
    return posterior

def gradients(theta):
    grads = bayes_model.get_gradients(theta)
    # print("grads {}".format(grads))
    return grads
#-----------------------------------------------------------------------------


# define a theano Op for our likelihood function
class LogLikeWithGrad(tt.Op):
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        theta, = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]


class LogLikeGrad(tt.Op):
    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        theta, = inputs

        # calculate gradients
        grads = gradients(theta)

        outputs[0][0] = grads



#-------

# create our Op
logl = LogLikeWithGrad(my_loglike)
# use PyMC3 to sampler from log-likelihood
with pm.Model() as opmodel:
    # for n in range(ndim):
    #     params.append(pm.Uniform('d_{}'.format(n), lower=-2., upper=10.))
    # theta = tt.as_tensor_variable(params)
    d1 = pm.Uniform('d1', lower=-1., upper=1.)
    d2 = pm.Uniform('d2', lower=-1., upper=1.)
    theta = tt.as_tensor_variable([d1, d2])

    # use a DensityDist
    pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})

    trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True)

# plot the traces
_ = pm.traceplot(trace)
plt.show()

# put the chains in an array (for later!)
# samples_pymc3_2 = np.vstack((trace['m'], trace['c'])).T