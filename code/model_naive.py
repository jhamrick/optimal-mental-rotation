import numpy as np


class NaiveModel(object):

    def __init__(self, x, y, verbose=False):
        """Initialize the linear interpolation object.

        Parameters
        ----------
        x : numpy.ndarray
            Vector of x values
        y : numpy.ndarray
            Vector of (actual) y values

        """

        # true x and y values for the likelihood
        self.x = x.copy()
        self.y = y.copy()
        # print fitting information
        self.verbose = verbose

    def fit(self, iix):
        """Fit the likelihood function.

        Parameters
        ----------
        iix : numpy.ndarray
            Integer array of indices corresponding to the "given" x and y data

        """

        # input data -- wrap around, so the interpolation works correctly
        self.xi = np.concatenate([self.x[iix], self.x[iix][[0]] + 2*np.pi])
        self.yi = np.concatenate([self.y[iix], self.y[iix][[0]]])

        self.mean = np.interp(self.x, self.xi, self.yi)

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


class HillClimbingSearch(object):

    def __init__(self, li, step, verbose=False):
        self.li = li
        self.step = step
        self.verbose = verbose

    def __iter__(self):
        Sr = self.li.y
        steps = np.arange(0, self.li.x.size, self.step)
        iix = [0]
        curr = 0
        scurr = steps[curr]

        if self.verbose:
            print "Starting at R=%d" % scurr
        yield scurr

        while True:
            next = curr + 1
            prev = curr - 1
            snext = steps[next]
            sprev = steps[prev]

            if next not in iix:
                if self.verbose:
                    print "R=% 3s degrees  S(X_b, X_R)=%f" % (snext, Sr[snext])
                iix.append(next)
                yield snext
            if prev not in iix:
                if self.verbose:
                    print "R=% 3s degrees  S(X_b, X_R)=%f" % (sprev, Sr[sprev])
                iix.append(prev)
                yield sprev

            if Sr[snext] > Sr[scurr]:
                curr = next
            elif Sr[sprev] > Sr[scurr]:
                curr = prev
            else:
                break

            scurr = steps[curr]
            if self.verbose:
                print "Pick R=%d" % scurr

        if self.verbose:
            print "Done"

    def sample(self):
        iix = sorted([i for i in self])
        return iix
