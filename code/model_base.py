import numpy as np
import scipy.misc
import util


class Model(object):

    def __init__(self, Xa, Xb, Xm, R, **opt):
        """Initialize the mental rotation model.

        Parameters
        ----------
        Xa : numpy.ndarray
            Stimulus A
        Xb : numpy.ndarray
            Stimulus B
        Xm : numpy.ndarray
            Array of rotations of stimulus A, corresponding to R
        R : numpy.ndarray
            Vector of all possible rotations, in radians. Each index i
            should correspond to an angle i in degrees.
        **opt : Model options (see below)

        Model options
        -------------
        verbose : bool (default=False)
            Print information during the modeling process
        scale : float (default=1)
            Scale of the data
        dr : int (default=10)
            Angle of rotation, in degrees, between sequential mental images
        sigma_s : float (default=0.2)
            Standard deviation in similarity function
        prior_R : float or numpy.ndarray (default=1/2pi)
            Prior over rotations

        """

        # default options
        default_opt = {
            'verbose': False,
            'scale': 1,
            'dr': 10,
            'sigma_s': 0.2,
            'prior_R': np.ones_like(R) / (2*np.pi),
        }
        # self.opt was defined by a subclass
        if hasattr(self, 'opt'):
            default_opt.update(self.opt)
        # user overrides
        default_opt.update(opt)
        self.opt = default_opt

        # stimuli
        self.Xa = Xa.copy()
        self.Xb = Xb.copy()
        self.Xm = Xm.copy()

        # all possible rotations
        self.R = R.copy()
        rot = np.round(np.arange(0, self.R.size - 1, self.opt['dr']))
        self._rotations = np.round(rot).astype('i8')
        # compute similarities
        self._S_scale = self.Xa.shape[0] - 1
        self._S_scale /= (2 * np.pi * self.opt['sigma_s']) * self.opt['scale']
        self.S = np.array([self.similarity(X) for X in self.Xm])
        self.S *= self._S_scale

        # prior over stimuli
        self.p_Xa = self.prior_X(Xa)
        self.p_Xb = self.prior_X(Xb)

        # joint of h0
        self.p_XaXb_h0 = self.p_Xa * self.p_Xb

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
        if self.opt.get('kernel', None) == 'gaussian':
            self.sample(self.R.size - 1)

    def sample(self, r):
        assert isinstance(r, int)
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
        verbose = self.opt['verbose']
        util.print_line(verbose=verbose)
        for i in self:
            util.print_line(verbose=verbose)
        if self.Ri is None or self.Si is None:
            self.fit()
            self.integrate()
        util.print_line(char="#", verbose=verbose)
        self.print_Z()
        self.ratio_test()

    def print_Z(self, level=-1):
        if self.Z_var == 0:
            self.debug("Z = %f" % (self.Z_mean), level=level)
        else:
            std = np.sqrt(self.Z_var)
            sf = self.opt['stop_factor']
            mean = self.Z_mean
            lower = mean - sf*std
            upper = mean + sf*std
            self.debug("Z = %f  [%f, %f]" % (mean, lower, upper),
                       level=level)

    def debug(self, msg, level=0):
        if self.opt['verbose'] > level:
            print ("  "*level) + msg

    def likelihood_ratio(self):
        std = 0 if self.Z_var == 0 else np.sqrt(self.Z_var)
        sf = self.opt['stop_factor']
        vals = [self.Z_mean, self.Z_mean - sf*std, self.Z_mean + sf*std]
        ratios = []
        for val in vals:
            p_XaXb_h1 = self.p_Xa * val / self._S_scale
            ratios.append(p_XaXb_h1 / self.p_XaXb_h0)
        return tuple(ratios)

    def ratio_test(self, level=-1):
        ratios = self.likelihood_ratio()
        self.debug("p(Xa, Xb | h1) / p(Xa, Xb | h0) = %f  [%f, %f]" % ratios,
                   level=level)
        if ratios[1] > 1:
            self.debug("--> Hypothesis 1 is more likely", level=level)
            return 1
        elif ratios[2] < 1 and ratios[0] > 0:
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
        Sigma = np.eye(D) * self.opt['sigma_s']
        invSigma = np.eye(D) * (1. / self.opt['sigma_s'])
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
