import numpy as np
from numpy import dot, trapz
from snippets.safemath import MIN_LOG

from periodic_kernel import PeriodicKernel as kernel
from gaussian_process import GP


class Evidence(object):

    def __init__(self, bq):
        self.gamma = 1.0
        self.opt = bq.opt
        self.p_R = self.opt['prior_R']

        self.xs = bq.Ri
        self.x_sc = bq.Rc
        self.x = bq.R
        self.l_s = bq.Si
        self.tl_s = self.log_transform(bq.Si)
        # XXX: need to actually calculate this right... fine for now
        # because gamma=1
        self.delta_tl_sc = bq.Dc

        self.kern_l = kernel(*bq.theta_S)
        self.gp_l = GP(self.kern_l, self.xs, self.l_s, self.x)

        self.kern_tl = kernel(*bq.theta_S)
        self.gp_tl = GP(self.kern_tl, self.xs, self.tl_s, self.x)

        self.kern_del = kernel(*bq.theta_S)
        self.gp_del = GP(self.kern_del, self.x_sc, self.delta_tl_sc, self.x)

    def log_transform(self, x):
        return np.log((x / self.gamma) + 1)

    def E(self, f, axis=-1):
        pfx = f * self.p_R
        m = trapz(pfx, self.x, axis=axis)
        return m

    ##################################################################
    # Mean

    @property
    def _mean(self):
        # Mean of int l(x) p(x) dx given l_s
        minty_l = self.E(self.gp_l.m)
        return minty_l

    @property
    def _mean_correction(self):
        # Mean of int delta(x) l(x) p(x) dx given l_s and delta_tl_sc
        E_mdel_ml = self.E(self.gp_del.m * self.gp_l.m)
        # Mean of int delta(x) p(x) dx given l_s and delta_tl_sc
        E_mdel = self.E(self.gp_del.m)
        # We need to multiply by gamma because we actually want to calculate
        #   E[ (1 / gamma) (m_l + gamma) * m_del ]
        # = (1 / gamma) (E[m_l * m_del] + gamma*E[m_del])
        # We'll rescale by gamma later.
        return E_mdel_ml + self.gamma * E_mdel

    @property
    def mean(self):
        mean_ev = self._mean + self._mean_correction
        # sanity check
        if mean_ev < 0:
            print 'mean of evidence negative'
            print 'mean of evidence: %s' % mean_ev
            mean_ev = self._mean
        return mean_ev

    ##################################################################
    # Variance

    @property
    def dm_dw(self):
        x, xs = self.x, self.xs
        inv_Kxx = self.gp_tl.inv_Kxx
        Kxox = self.gp_tl.Kxox
        dKxox_dw = self.gp_tl.K.dK_dw(x, xs)
        dKxx_dw = self.gp_tl.K.dK_dw(xs, xs)
        inv_dKxx_dw = dot(inv_Kxx, dot(dKxx_dw, inv_Kxx))
        dm_dw = (dot(dKxox_dw, dot(inv_Kxx, self.tl_s)) -
                 dot(Kxox, dot(inv_dKxx_dw, self.tl_s)))
        return dm_dw

    @property
    def Cw(self):
        """The variances of our posteriors over our input scale. We assume the
        covariance matrix has zero off-diagonal elements; the posterior
        is spherical.

        """
        # H_theta is the diagonal of the hessian of the likelihood of
        # the GP over the log-likelihood with respect to its log input
        # scale.
        H_theta = self.gp_tl.d2lh_dtheta2()
        Cw = -1. / H_theta[1, 1]
        return Cw

    @property
    def _var(self):
        x, xs, p_R = self.x, self.xs, self.p_R
        gamma = self.gamma
        C_tl = self.gp_tl.C
        m_l = self.gp_l.m

        # int int C_(tl|s) p(x) p(x') dx dx'
        E_Ctl = self.E(self.E(C_tl))

        # int int p(x) p(x') C_(tl|s)(x, x') m_(l|s)(x') dx dx'
        E_Ctl_ml = self.E(self.E(C_tl * m_l[None, :]))

        # int dx p(x) int dx' p(x') m_(l|s)(x) C_(tl|s)(x, x') m_(l|s)(x')
        E_ml_Ctl_ml = self.E(self.E(C_tl * m_l[None, :] * m_l[:, None]))

        # Again, we need to mutliply by gamma because:
        #   E[E[(1/gamma**2) (m_l + gamma) * C_tl * (m_l + gamma)]]
        # = (1 / gamma**2) * (
        #      gamma**2 * E[E[C_tl]] +
        #      2*gamma * E[E[m_l * C_tl]] +
        #      E[E[m_l * C_tl * m_l]])
        var_ev = (gamma**2 * E_Ctl) + (2*gamma * E_Ctl_ml) + E_ml_Ctl_ml
        return var_ev

    @property
    def _var_correction(self):
        dm_dw, Cw = self.dm_dw, self.Cw
        m_l = self.gp_l.m

        E_ml_dmtldw = self.E(m_l * dm_dw)
        E_dmtldw = self.E(dm_dw)

        correction = Cw * (E_ml_dmtldw + self.gamma*E_dmtldw) ** 2
        return correction

    @property
    def var(self):
        var_ev = self._var + self._var_correction
        # sanity check
        if var_ev < 0:
            print 'variance of evidence negative'
            print 'variance of evidence: %s' % var_ev
            var_ev = np.exp(MIN_LOG)
        return var_ev
