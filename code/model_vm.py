import numpy as np
import circstats as circ
import scipy.optimize as opt

from model_base import Model
from search import hill_climbing


class VonMisesModel(Model):

    def __init__(self, *args, **kwargs):
        """See Model.__init__

        Additional model options
        ------------------------
        ntry : int (default=10)
           Number of times to run optimization function

        """

        # default options
        self.opt = {
            'ntry': 10
        }

        super(VonMisesModel, self).__init__(*args, **kwargs)
        self._icurr = 0
        self._ilast = None

    @staticmethod
    def _mse(theta, x, y):
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

    def next(self):
        """Sample the next point."""

        self.debug("Finding next sample")

        icurr = hill_climbing(self)
        if icurr is None:
            raise StopIteration

        self._ilast = self._icurr
        self._icurr = icurr

    def fit(self):
        """Fit the likelihood function."""

        self.debug("Fitting likelihood")

        self.ix = sorted(self.ix)
        self.Ri = self.R[self.ix]
        self.Si = self.S[self.ix]

        args = np.empty((self.opt['ntry'], 3))
        fval = np.empty(self.opt['ntry'])

        for i in xrange(self.opt['ntry']):
            # randomize starting parameter values
            t0 = np.random.uniform(0, 2*np.pi)
            k0 = np.random.gamma(2, 2)
            z0 = np.random.uniform(0, np.max(np.abs(self.Si))*2)
            p0 = (t0, k0, z0)

            # run mimization function
            popt = opt.minimize(
                fun=self._mse,
                x0=p0,
                args=(self.Ri, self.Si),
                method="L-BFGS-B",
                bounds=((0, 2*np.pi), (1e-8, None), (1e-8, None))
            )

            # get results of the optimization
            success = popt['success']
            message = popt['message']

            if not success:
                args[i] = np.nan
                fval[i] = np.inf
                self.debug("Failed: %s" % message, level=3)
            else:
                args[i] = abs(popt['x'])
                fval[i] = popt['fun']
                self.debug("MSE(%s) = %f" % (args[i], fval[i]), level=3)

        # choose the parameters that give the smallest MSE
        best = np.argmin(fval)
        if np.isinf(fval[best]) and np.sign(fval[best]) > 0:
            print args[best], fval[best]
            raise RuntimeError("Could not find parameter estimates")

        self.theta = args[best]
        self.S_mean = self.theta[2] * circ.vmpdf(self.R, *self.theta[:2])
        self.S_var = np.zeros(self.S_mean.shape)

        self.debug("Best parameters: %s" % self.theta, level=2)

    def integrate(self):
        """Compute the mean and variance of Z:

        $$Z = \int S(X_b, X_R)p(R) dR$$

        """

        if self.S_mean is None or self.S_var is None:
            raise RuntimeError(
                "S_mean or S_var is not set, did you call self.fit first?")

        self.Z_mean = sum(self.pR * self.S_mean)
        self.Z_var = 0
        self.print_Z(level=0)
