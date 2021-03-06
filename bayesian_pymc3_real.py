import theano.tensor as tt
import theano
import pymc3 as pm
import numpy as np
import arviz as az
from battaglia_full import Dens2bBatt
from battaglia_plus_bayes import DensErr
import matplotlib.pyplot as plt
import corner

az.style.use('arviz-darkgrid')

z = 9
one_d_size = ndim = 16
mu, sigma_T, sigma_D = 0, 1, 1
ones = np.ones(one_d_size)
cov = cov_prior = np.diag(ones)
cov *= sigma_T
cov_likelihood = cov ** 2

cov_prior = np.copy(cov)

for i in range(np.shape(cov_prior)[0]):
    if i != (np.shape(cov_prior)[0] - 1):
        cov_prior[i][i + 1] = 0.05
        if i < (np.shape(cov_prior)[0] - 2):
            cov_prior[i][i + 2] = 0.05
    if i != 0:
        cov_prior[i][i - 1] = 0.15
        if i > 1:
            cov_prior[i][i - 2] = 0.05

# correlated densities
x = np.random.randn(one_d_size)
L = np.linalg.cholesky(cov_prior)
correlated_vars = np.dot(L, x)
one_rho = log_norm_correlated_density = np.exp(correlated_vars) - 1
dens2Tb = Dens2bBatt(one_rho, 1, z, one_d=True)
one_Tb = dens2Tb.temp_brightness



# LOAD IN DENSITIES FROM 21cmFAST
# dir = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/21cmFASTData/21cmFASTBoxes_{}/PerturbedField_*".format(z)
# hf = h5py.File(glob.glob(dir)[0], 'r')
# line_rho = hf["PerturbedField/density"][:, 0, 0]
# dens2Tb = Dens2bBatt(line_rho, 1, z, one_d=True)
# line_Tb = one_Tb = dens2Tb.temp_brightness
# one_rho = actual_rho = line_rho[:one_d_size]
# one_Tb = actual_Tb = line_Tb[:one_d_size]

np.save("/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/CORR_DATA/one_rho_z{}_size{}.npy".format(z, one_d_size), one_rho)
np.save("/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/CORR_DATA/one_Tb_z{}_size{}.npy".format(z, one_d_size), one_Tb)
print("one_rho {}".format(one_rho))
print("one_Tb {}".format(one_Tb))

# log likelihood function
test1D = DensErr(z, [0,0], one_Tb, cov_likelihood, cov_prior, sigma_D, sigma_T, epsilon=0, actual_rhos=[0,0], pyMC3=True, no_prior_uncorr=True, log_norm=False)

def gauss_likelihood(prm):
    """
    A Gaussian log-likelihood function for a model with parameters given in prm
    """
    print("test1D.numeric_posterior(prm) {}".format(test1D.numeric_posterior(prm)))
    logp, grads = test1D.numeric_posterior(prm)
    # print("grads {}".format(grads))
    return logp

def gradients(theta):
    logp, grads = test1D.numeric_posterior(theta)
    # print(grads)
    return grads

# define a theano Op for our gradient
class LogLikeGrad(tt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self):
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

    def perform(self, node, inputs, outputs):
        theta, = inputs
        # calculate gradients
        grads = gradients(theta) # TODO sigma is wrong
        outputs[0][0] = grads

# define a theano Op for our likelihood function
class LogLikeWithGrad(tt.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in

        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad()


    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        prm, = inputs  # this will contain my variables
        # call the log-likelihood function
        logl = self.likelihood(prm)
        outputs[0][0] = np.array(logl) # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        theta, = inputs  # our parameters
        return [g[0]*self.logpgrad(theta)]

ndraws = 200  # number of draws from the distribution
nburn = 100 # number of "burn-in points" (which we'll discard)


logl = LogLikeWithGrad(gauss_likelihood)


# test the gradient Op by direct call
theano.config.compute_test_value = "ignore"
theano.config.exception_verbosity = "high"

with pm.Model():
    params = []
    starts = {}
    starts_arr = []

    for i in range(one_d_size):
        # param = one_rho[i] + np.random.randn()
        # params.append(pm.Uniform('d_{}'.format(i), lower=-1, upper=3, ))
        params.append(pm.Bound(pm.Normal, lower=-1.0)('d_{}'.format(i), mu=0.0, sigma=4.0))
        start = 0
        if one_Tb[i] == 0:
            start = one_rho[i] + np.random.randn()
            starts['d_{}'.format(i)] = start
        starts_arr.append(start)

    np.save("/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/CORR_DATA/inital_{}_{}.npy".format(z, one_d_size), starts_arr)

    prm = tt.as_tensor_variable(params)

    # use a DensityDist (use a lamdba function to "call" the Op)
    pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': prm})

    trace = pm.sample(ndraws, tune=nburn, cores=4, start=starts)
    pm.save_trace(trace, directory="/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/T2D2 Model/STAT_DATA/TRACES/z_{}_{}.trace".format(z, one_d_size), overwrite=True)

# samples_pymc3 = np.vstack((trace['d_0'], trace['d_1'], trace['d_2'], trace['d_3'], trace['d_4'], trace['d_5'], trace['d_6'], trace['d_7'])).T
# fig = corner.corner(samples_pymc3, labels=["d_0", "d_1", "d_2", "d_3", "d_4", "d_5", "d_6", "d_7"])
# plt.show()

