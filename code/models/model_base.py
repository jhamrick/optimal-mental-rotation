import numpy as np
import scipy.misc
import tools
import copy
import warnings
from itertools import izip


class Model(object):

    # default options
    default_opt = {
        'verbose': False,
        'scale': 1,
        'dr': 10,
        'sigma_s': 0.2,
        'prior_R': (0, 1),
    }

    def __init__(self, *args, **opt):
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

        if len(args) == 1:
            self._copyfrom(args[0])
            return

        Xa, Xb, Xm, R = args

        # self.opt was defined by a subclass
        opts = self.default_opt.copy()
        opts.update(getattr(self, 'opt', {}))
        opts.update(opt)
        self.opt = opt

        # copy given arguments
        self.Xa = Xa.copy()
        self.Xb = Xb.copy()
        self.Xm = Xm.copy()
        self.R = R.copy()

        # compute prior probabilities of angles
        mu, var = self.opt['prior_R']
        self.pR = scipy.stats.norm.pdf(self.R, mu, np.sqrt(var))

        # compute the prior over stimuli
        self.p_Xa = self.prior_X(Xa)
        self.p_Xb = self.prior_X(Xb)

        # compute joint of h0
        self.p_XaXb_h0 = self.p_Xa * self.p_Xb

        # compute similarities
        self._S_scale = (
            (self.Xa.shape[0] - 1) /
            (2 * np.pi * self.opt['sigma_s'] * self.opt['scale']))
        self.S = np.array(
            [tools.similarity(self.Xb, X, self.opt) for X in self.Xm])
        self.S *= self._S_scale

        # possible sample points
        self._all_samples = {}
        deg = np.round(np.degrees(self.R)).astype('i8') % 360
        for R, S in izip(deg, self.S):
            if R in self._all_samples:
                warnings.warn(
                    "Warning: rotation %d already exists, "
                    "overwriting" % R, RuntimeWarning)
            self._all_samples[R] = S

        self.initialize()

    def initialize(self):
        # sampled R and S values
        self._sampled = np.array([])
        self.Ri = np.array([])
        self.Si = np.array([])

        # mean and variance of S
        self.S_mean = None
        self.S_var = None

        # mean and variance of Z
        self.Z_mean = None
        self.Z_var = None

        # we get the first observation for free
        self.sample(0)

    def _copyfrom(self, other):
        self.opt = copy.deepcopy(other.opt)

        self.Xa = other.Xa.copy()
        self.Xb = other.Xb.copy()
        self.Xm = other.Xm.copy()
        self.p_Xa = other.p_Xa
        self.p_Xb = other.p_Xb
        self.p_XaXb_h0 = other.p_XaXb_h0
        self.R = other.R.copy()
        self.S = other.S.copy()
        self._S_scale = other._S_scale
        self._all_samples = other._all_samples.copy()

        # these are defined by `initialize`
        self._sampled = other._sampled.copy()
        self.Ri = other.Ri.copy()
        self.Si = other.Si.copy()
        self.S_mean = other.S_mean
        self.S_var = other.S_var
        self.Z_mean = other.Z_mean
        self.Z_var = other.Z_var

    def copy(self):
        cls = type(self)
        return cls(self)

    def __iter__(self):
        return self

    def _get_S(self, d):
        # r will be in radians; convert it to the nearest round degree
        dw = d % 360
        S = self._all_samples[dw]
        return S

    @property
    def curr_val(self):
        """Most recently sampled R and S"""
        R = np.radians(self._sampled[-1])
        d = int(np.round(np.degrees(R)))
        S = self._get_S(d)
        return R, S

    @property
    def num_samples_left(self):
        all_samples = np.array(self._all_samples.keys())
        sampled = self._sampled % 360
        remaining = np.setdiff1d(all_samples, sampled)
        return remaining.size

    def observed(self, R):
        """Check whether R has already been sampled or not."""
        if R is None:
            return False
        # r will be in radians; convert it to the nearest round degree
        dw = int(np.round(np.degrees(R))) % 360
        sampled = self._sampled % 360
        return (dw == sampled).any()

    def sample(self, R):
        d = int(np.round(np.degrees(R)))
        S = self._get_S(d)
        if not self.observed(R):
            self.Ri = np.append(self.Ri, R % (2*np.pi))
            self.Si = np.append(self.Si, S)
        # add to the list of sampled regardless of whether we've
        # already seen it, so we can keep track of the sequence of
        # rotations
        self._sampled = np.append(self._sampled, d)
        self.debug("R=% 3s degrees  S(X_b, X_R)=%f" % (d, S), level=1)
        return S

    def next_val(self):
        if self._sampled.size < 2:
            # pick a random direction
            direction = (np.random.randint(0, 2) * 2) - 1
        else:
            direction = np.sign(self._sampled[-1] - self._sampled[-2])
        R = np.radians(self._sampled[-1])
        R_next = R + (direction * self.opt['dr'])
        S_next = self.sample(R_next)
        return R_next, S_next

    def run(self):
        verbose = self.opt['verbose']
        tools.print_line(verbose=verbose)
        for i in self:
            tools.print_line(verbose=verbose)
        tools.print_line(char="#", verbose=verbose)
        self.print_Z()
        self.ratio_test()

    def print_Z(self, level=-1):
        if not self.Z_var:
            self.debug("Z = %f" % (self.Z_mean), level=level)
        else:
            sf = self.opt['stop_factor']
            mean = self.Z_mean
            std = np.sqrt(self.Z_var)
            lower = mean - sf*std
            upper = mean + sf*std
            self.debug("Z = %f  [%f, %f]" % (mean, lower, upper),
                       level=level)

    def debug(self, msg, level=0):
        if self.opt['verbose'] > level:
            print ("  "*level) + msg

    def likelihood_ratio(self):
        sf = self.opt['stop_factor']
        std = 0 if not self.Z_var else np.sqrt(self.Z_var)
        vals = [self.Z_mean, self.Z_mean - sf*std, self.Z_mean + sf*std]
        ratios = [
            self.p_Xa * val / (self._S_scale * self.p_XaXb_h0)
            for val in vals
        ]
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

    def integrate(self):
        """Compute the mean and variance of Z:

        $$Z = \int S(X_b, X_R)p(R) dR$$

        This is a dumb brute force way of doing this, by multiplying
        S_mean by the prior and then using the trapezoidal rule to
        approximate the integral.

        """

        if self.S_mean is None or self.S_var is None:
            raise RuntimeError(
                "S_mean or S_var is not set, did you call self.fit first?")

        self.Z_mean = np.trapz(self.pR * self.S_mean, self.R)
        self.Z_var = 0
