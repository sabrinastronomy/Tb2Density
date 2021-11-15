import theano.tensor as tt
import theano
import pymc3 as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import corner

az.style.use('arviz-darkgrid')

# log likelihood function
def gauss_likelihood(prm, Tarr, cov=[]):
    """
    A Gaussian log-likelihood function for a model with parameters given in prm
    """
    # rho_vec = np.vstack(np.asarray(prm))
    # T_vec = np.vstack(Tarr)
    rho_vec = prm
    T_vec = Tarr

    # c = np.linalg.det(cov)
    # exp_1 = (-0.5) * (T_vec - (rho_vec ** 2)).T
    # exp_1 = np.dot(exp_1, np.linalg.inv(cov))
    # exp_1 = np.dot(exp_1, (T_vec - (rho_vec ** 2))).flatten()
    # prefactor_likelihood = np.log(1 / (np.sqrt(2 * np.pi) * c) ** len(Tarr))

    # uncorrelated likelihood
    trans_X = (-0.5) * (T_vec - (rho_vec ** 2)).T
    trans_X_C = np.dot(trans_X, np.linalg.inv(cov))
    trans_X_C_X = np.dot(trans_X_C, (T_vec - (rho_vec ** 2)))
    print("trans_X_C_X {}".format(trans_X_C_X))

    c = np.linalg.det(cov)
    print("det cov likelihood {}".format(c))
    prefactor_likelihood = np.log(1 / (np.sqrt(2 * np.pi) * c) ** len(Tarr))
    print("prefactor_likelihood {}".format(prefactor_likelihood))

    prefactor = prefactor_likelihood
    logp = prefactor + (trans_X_C_X)
    return logp

def gradients(theta, T_meas, cov, toy=True):
    if toy:
        alpha = 1 # placeholder
        densities = theta # what we're sampling
        sigma_2 = cov[0][0] # sigma^2, one element of cov matrix
        grads = densities**2 * (1/sigma_2)*(-alpha*densities**2 + T_meas) # calculated gradients for each density value
        return grads

# define a theano Op for our gradient
class LogLikeGrad(tt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, Tarr, cov=[]):
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
        self.data = Tarr
        self.cov = cov

    def perform(self, node, inputs, outputs):
        theta, = inputs

        # calculate gradients
        grads = gradients(theta, self.data, self.cov) # TODO sigma is wrong
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

    def __init__(self, loglike, Tarr, cov=[]):
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
        self.data = Tarr
        self.cov = cov
        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.data, self.cov)


    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        prm, = inputs  # this will contain my variables
        # call the log-likelihood function
        logl = self.likelihood(prm, self.data, self.cov)
        outputs[0][0] = np.array(logl) # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        theta, = inputs  # our parameters
        return [g[0]*self.logpgrad(theta)]

ndraws = 100  # number of draws from the distribution
nburn = 10   # number of "burn-in points" (which we'll discard)

ndim = 3
Tarr = [10, 10, 10]
sigma_d = 10
sigma_a = 0.2

# cov = np.abs(datasets.make_spd_matrix(ndim)*10) # TODO: change this so change the prior, don't generate this randomly
cov = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# create our Op
logl = LogLikeWithGrad(gauss_likelihood, Tarr, cov=cov)


# test the gradient Op by direct call
theano.config.compute_test_value = "ignore"
theano.config.exception_verbosity = "high"
#
# var = tt.dvector()
# test_grad_op = LogLikeGrad(Tarr, cov)
# test_grad_op_func = theano.function([var], test_grad_op(var))
# grad_vals = test_grad_op_func([1, 2, 3])

# print('Gradient returned by "LogLikeGrad": {}'.format(grad_vals))

with pm.Model():
    params = []
    # params.append(pm.Normal('alpha', mu=0, sigma=sigma_a))
    params.append(pm.Normal('d_0', mu=5, sigma=1))
    params.append(pm.Normal('d_1', mu=5, sigma=1))
    params.append(pm.Normal('d_2', mu=5, sigma=1))
    # for n in range(ndim):
    #     params.append(pm.Normal('d_{}'.format(n), mu=0, sigma=sigma_d))
    prm = tt.as_tensor_variable(params)

    # use a DensityDist (use a lamdba function to "call" the Op)
    pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': prm})
    trace = pm.sample(ndraws, tune=nburn, cores=1)
    # samples_pymc3 = np.vstack(trace['d_0']).T

    samples_pymc3 = np.vstack((trace['d_0'], trace['d_1'], trace['d_2'])).T
# _ = pm.traceplot(trace)

fig = corner.corner(samples_pymc3, labels=["d_0", "d_1", "d_2"])
# fig = corner.corner(trace['d_0'])
plt.show()

