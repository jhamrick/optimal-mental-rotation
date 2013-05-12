import numpy as np
import scipy.misc
import util


class Model(object):

    def __init__(self, R, S, dr, pR, **opt):
        """Initialize the linear interpolation object.

        Parameters
        ----------
        R : numpy.ndarray
            Vector of all possible rotations, in radians. Each index i
            should correspond to an angle i in degrees.
        S : numpy.ndarray
            Vector of all values of the similarity function S. Each
            index i should correspond to an angle i in degrees.
        dr : int
            Angle of rotation in between sequential mental images
        pR : numpy.ndarray
            Probability vector for all values of R. Should have the same
            dimensions as S.
        **opt : Model options (see below)

        Model options
        -------------
        verbose : bool (default=False)
            Print information during the modeling process
        scale : float (default=1)
            Scale of the data

        """

        # default options
        default_opt = {
            'verbose': False,
            'scale': 1,
        }
        # self.opt was defined by a subclass
        if hasattr(self, 'opt'):
            default_opt.update(self.opt)
        # user overrides
        default_opt.update(opt)
        self.opt = default_opt

        # step size
        self.dr = dr

        # prior over angles
        self.pR = pR

        # x and y values
        self.R = R.copy()
        self.S = S.copy()
        self._rotations = np.arange(0, self.R.size, self.dr)

        # samples
        self.ix = []
        # sampled R and S values
        self.Ri = None
        self.Si = None

        # mean and variance of S
        self.S_mean = None
        self.S_var = None

        # mean and variance of Z
        self.Z_mean = None
        self.Z_var = None

        # we get the first (and last, because it's circular data)
        # observation for free
        self.sample(0)

    def sample(self, r):
        S = self.S[r]
        if r not in self.ix:
            self.ix.append(r)
            self.Ri = None
            self.Si = None
            self.debug("R=% 3s degrees  S(X_b, X_R)=%f" % (r, S), level=1)
        return S

    def __iter__(self):
        return self

    def run(self):
        for i in self:
            pass

        if self.Ri is None or self.Si is None:
            self.fit()
            self.integrate()

        if not self.opt['verbose']:
            self.print_Z()

    def print_Z(self, level=-1):
        if self.Z_var == 0:
            self.debug("Z = %f" % (self.Z_mean), level=level)
        else:
            std = np.sqrt(self.Z_var)
            mean = self.Z_mean
            lower = mean - 2*std
            upper = mean + 2*std
            self.debug("Z = %f  [%f, %f]" % (mean, lower, upper),
                       level=level)

    def debug(self, msg, level=0):
        if self.opt['verbose'] > level:
            print ("  "*level) + msg

    def likelihood_ratio(self):
        std = 0 if self.Z_var == 0 else np.sqrt(self.Z_var)
        vals = [self.Z_mean, self.Z_mean - 2*std, self.Z_mean + 2*std]
        ratios = []
        for val in vals:
            p_XaXb_h1 = self.p_Xa * val / self.opt['scale']
            ratios.append(p_XaXb_h1 / self.p_XaXb_h0)
        return tuple(ratios)

    def ratio_test(self, level=-1):
        ratios = self.likelihood_ratio()
        self.debug("p(Xa, Xb | h1) / p(Xa, Xb | h0) = %f  [%f, %f]" % ratios,
                   level=level)
        if ratios[1] > 1:
            self.debug("--> Hypothesis 1 is more likely", level=level)
            return 1
        elif ratios[2] < 1 and ratios[2] > 0:
            self.debug("--> Hypothesis 0 is more likely", level=level)
            return 0
        else:
            self.debug("--> Undecided", level=level)
            return -1

    @staticmethod
    def prior_X(X):
        # the beginning is the same as the end, so ignore the last vertex
        n = X.shape[0] - 1
        # n points picked at random angles around the circle
        log_pangle = -np.log(2*np.pi) * n
        # random radii between 0 and 1
        radius = 1
        log_pradius = -np.log(radius) * n
        # number of possible permutations of the points
        log_pperm = np.log(scipy.misc.factorial(n))
        # put it all together
        p_X = np.exp(log_pperm + log_pangle + log_pradius)
        return p_X

    def similarity(self, X):
        """Computes the similarity between sets of vertices `X0` and `X1`."""
        # the beginning is the same as the end, so ignore the last vertex
        x0 = self.Xb[:-1]
        x1 = X[:-1]
        # number of points and number of dimensions
        n, D = x0.shape
        # covariance matrix
        Sigma = np.eye(D) * self.opt['sigma']
        invSigma = np.eye(D) * (1. / self.opt['sigma'])
        # iterate through all permutations of the vertices -- but if two
        # vertices are connected, they are next to each other in the list
        # (or on the ends), so we really only need to cycle through n
        # orderings
        e = np.empty(n)
        for i in xrange(n):
            idx = np.arange(i, i+n) % n
            d = x0 - x1[idx]
            e[i] = -0.5 * np.sum(np.dot(d, invSigma) * d)
        # constants
        Z0 = (D / 2.) * np.log(2 * np.pi)
        Z1 = 0.5 * np.linalg.slogdet(Sigma)[1]
        # overall similarity, marginalizing out order
        S = np.sum(np.exp(e + Z0 + Z1 - np.log(n)))
        return S
