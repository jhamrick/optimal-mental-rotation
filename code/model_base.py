import numpy as np


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

        """

        # default options
        default_opt = {
            'verbose': False
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
            if self.opt['verbose']:
                print "R=% 3s degrees  S(X_b, X_R)=%f" % (r, S)
        return S

    def __iter__(self):
        return self

    def run(self):
        for i in self:
            if self.opt['verbose']:
                self.fit()
                self.integrate()
                print "mu_Z  = %f" % self.m_Z
                print "var_Z = %f" % self.V_Z

        if self.Ri is None or self.Si is None:
            self.fit()
            self.integrate()
