import numpy as np
import circstats as circ
import scipy.optimize as opt


class VonMisesModel(object):

    def __init__(self, x, y, ntry=10, verbose=False):
        """Initialize the likelihood estimator object.

        Parameters
        ----------
        x : numpy.ndarray
            Vector of x values
        y : numpy.ndarray
            Vector of (actual) y values
        ntry : int (default=10)
            Number of optimizations to run
        verbose : bool (default=False)
            Whether to print information during fitting

        """

        # true x and y values for the likelihood
        self.x = x.copy()
        self.y = y.copy()
        # number of optimization tries
        self.ntry = ntry
        # print fitting information
        self.verbose = verbose

    @staticmethod
    def mse(theta, x, y):
        """Computes the mean squared error of the Von Mises PDF.

        Parameters
        ----------
        theta : 3-tuple
            The Von Mises PDF parameters `thetahat`, `kappa`, and `z`
        x : numpy.ndarray with shape (n,)
            The given input values
        y : numpy.ndarray with shape (n,)
            The given output values

        Returns
        -------
        out : float
            The mean squared error of the PDF given the parameters

        """

        thetahat, kappa, z = theta
        pdf = np.log(z) + circ.vmlogpdf(x, thetahat, kappa)
        err = np.sum((y - np.exp(pdf)) ** 2)
        return err

    def fit(self, iix):
        """Fit the likelihood function.

        Parameters
        ----------
        iix : numpy.ndarray
            Integer array of indices corresponding to the "given" x and y data

        """

        # input data
        self.xi = self.x[iix].copy()
        self.yi = self.y[iix].copy()

        args = np.empty((self.ntry, 3))
        fval = np.empty(self.ntry)

        for i in xrange(self.ntry):
            # randomize starting parameter values
            t0 = np.random.uniform(0, 2*np.pi)
            k0 = np.random.gamma(2, 2)
            z0 = np.random.uniform(0, np.max(np.abs(self.yi))*2)
            p0 = (t0, k0, z0)

            # run mimization function
            popt = opt.minimize(
                fun=self,
                x0=p0,
                args=(self.xi, self.yi),
                method="L-BFGS-B",
                bounds=((0, 2*np.pi), (1e-8, None), (1e-8, None))
            )

            # get results of the optimization
            success = popt['success']
            message = popt['message']

            if not success:
                args[i] = np.nan
                fval[i] = np.inf
                if self.verbose:
                    print "Failed: %s" % message
            else:
                args[i] = abs(popt['x'])
                fval[i] = popt['fun']
                if self.verbose:
                    print "MSE(%s) = %f" % (args[i], fval[i])

        # choose the parameters that give the smallest MSE
        best = np.argmin(fval)
        if np.isinf(fval[best]) and np.sign(fval[best]) > 0:
            print args[best], fval[best]
            raise RuntimeError("Could not find parameter estimates")

        self.theta = args[best]
        self.mean = self.theta[2] * circ.vmpdf(self.x, *self.theta[:2])

    def integrate(self, px):
        """Compute the mean and variance of our estimate of the integral:

        $$Z = \int S(y|x)p(x) dx$$

        Where S(y|x) is the function being estimated by `self.fit`.

        Parameters
        ----------
        px : numpy.ndarray
            Prior probabilities over x-values

        Returns
        -------
        out : 2-tuple
            The mean and variance of the integral

        """

        m_Z = sum(px * self.mean)
        V_Z = 0
        return m_Z, V_Z
