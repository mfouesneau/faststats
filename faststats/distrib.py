""" Class Distribution

This class implements a distribution object that is defined by its pdf
(probability density function)

Interestingly, I could not find in numpy/scipy a class that could implement a
distribution just from its pdf.  The idea of such object is to be able to
compute statistics of this distribution without any pain.

Also this class implements basic operations such as + - / *, with scalars or
distributions, which comes handy when doing probabilities.  The normalization
is left to the user's decision but can be quickly done using the normalize()
method

Note that operations with scalars requires to write the distribution on the
left side of the operators.

Implementation notes:

    * the pdf is given by a linear interpolation of the samples,
    * the pdf's norm is given by a scipy.integrate.simps integration (fast and robust)
    * the cdf is given by the linear interpolation of the cumulative sum of the pdf samples.
    * percentiles are calculated directly by bracketing the cdf and from linear interpolations
"""


import inspect
import numpy as np
from scipy.integrate import simps


class Distribution(object):
    def __init__(self, x, pdf, name=None, *args, **kwargs):
        if len(x) != len(pdf):
            raise ValueError('x and pdf must have the same length')
        ind = np.argsort(x)
        self._pdf = np.asarray(pdf)[ind]
        self._x = np.asarray(x)[ind]
        self.norm = simps(self._pdf, self._x)
        self.name = name

    def pdf(self, x):
        """Probability density function"""
        return np.interp(x, self._x, self._pdf) / self.norm

    def cdf(self, x):
        """Cumulative distribution function"""
        xp = self._x
        fp = np.cumsum(self._pdf)
        return np.interp(x, xp, fp) / self.norm

    def sf(self, x):
        """Survival function = complementary CDF"""
        return 1. - self.cdf(x, err=False)

    def ppf(self, x):
        """Percentile point function (i.e. CDF inverse)"""
        data = self._x
        weights = self._pdf / self.norm
        percentiles = np.clip(x * 100., 0., 100.)
        # check if actually weighted percentiles is needed
        if weights is None:
            return np.percentile(data, list(percentiles))
        if np.equal(weights, 1.).all():
            return np.percentile(data, list(percentiles))

        # make sure percentiles are fractions between 0 and 1
        if not np.greater_equal(percentiles, 0.0).all():
            raise ValueError("Percentiles less than 0")
        if not np.less_equal(percentiles, 100.0).all():
            raise ValueError("Percentiles greater than 100")

        #Make sure data is in correct shape
        shape = np.shape(data)
        n = len(data)
        if (len(shape) != 1):
            raise ValueError("wrong data shape, expecting 1d")

        if len(weights) != n:
            raise ValueError("weights must be the same shape as data")
        if not np.greater_equal(weights, 0.0).all():
            raise ValueError("Not all weights are non-negative.")

        _data = np.asarray(data, dtype=float)

        if hasattr(percentiles, '__iter__'):
            _p = np.asarray(percentiles, dtype=float) * 0.01
        else:
            _p = np.asarray([percentiles * 0.01], dtype=float)

        _wt = np.asarray(weights, dtype=float)

        len_p = len(_p)
        sd = np.empty(n, dtype=float)
        sw = np.empty(n, dtype=float)
        aw = np.empty(n, dtype=float)
        o = np.empty(len_p, dtype=float)

        i = np.argsort(_data)
        np.take(_data, i, axis=0, out=sd)
        np.take(_wt, i, axis=0, out=sw)
        np.add.accumulate(sw, out=aw)

        if not aw[-1] > 0:
            raise ValueError("Nonpositive weight sum")

        w = (aw - 0.5 * sw) / aw[-1]

        spots = np.searchsorted(w, _p)
        for (pk, s, p) in zip(range(len_p), spots, _p):
            if s == 0:
                o[pk] = sd[0]
            elif s == n:
                o[pk] = sd[n - 1]
            else:
                f1 = (w[s] - p) / (w[s] - w[s - 1])
                f2 = (p - w[s - 1]) / (w[s] - w[s - 1])
                assert (f1 >= 0) and (f2 >= 0) and (f1 <= 1 ) and (f2 <= 1)
                assert abs(f1 + f2 - 1.0) < 1e-6
                o[pk] = sd[s - 1] * f1 + sd[s] * f2
        return o

    def isf(self, x):
        """Inverse survival function (Complementary CDF inverse)"""
        return 1. - self.ppf(x)

    @property
    def mean(self):
        e = (self._x * self._pdf).sum() / self._pdf.sum()
        return e

    @property
    def variance(self):
        m = self.mean
        e = ( (self._x - m) ** 2 * self._pdf).sum() / self._pdf.sum()
        return e

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def skew(self):
        return self.moment(3, reduced=True)

    @property
    def kurtosis(self):
        return self.moment(4, reduced=True)

    def moment(self, order, reduced=False):
        """Non-central moments"""
        X = self._x - self.mean
        if reduced is True:
            X /= self.std
        e = ( X ** order * self._pdf).sum() / self._pdf.sum()
        return e

    def rvs(self, N):
        """Random samples"""
        x = np.random.uniform(0., 1., N)
        return self.ppf(x)

    def normalize(self):
        """ Normalize the sampled pdf by its norm """
        self._pdf / self.norm

    def __add__(self, other):
        """ Sum of distributions """
        if np.isscalar(other):
            name = '{:s} + {}'.format(self.name, other)
            return Distribution(self._x, self._pdf + other, name=name)
        elif isinstance(other, Distribution):
            x0 = self._x
            pdf0 = self._pdf
            x1 = other._x
            pdf1 = other._pdf
            x = np.unique(np.hstack([x0, x1]))
            y0 = np.interp(x, x0, pdf0)
            y1 = np.interp(x, x1, pdf1)

            name = '({:s}) + ({:s})'.format(self.name, other.name)

            return Distribution(x, y0 + y1, name=name)
        elif hasattr(other, '__call__'):
            x0 = self._x
            y0 = self._pdf
            y1 = other(x0)

            n1 = getattr(other, '__name__', 'f(...)')
            if n1 == '<lambda>':
                t = inspect.getsource(other).replace(' ', '')[:-1]
                t = ''.join(t.split('lambda')[1:]).split(':')
                n1 = 'f({t[0]}) = {t[1]}'.format(t=t)
            name = '({:s}) + ({:s})'.format(self.name, n1)
            return Distribution(x0, y0 + y1, name=name)

    def __sub__(self, other):
        """ Subtract distribution """
        if np.isscalar(other):
            name = '{:s} + {}'.format(self.name, other)
            return Distribution(self._x, self._pdf + other, name=name)
        elif isinstance(other, Distribution):
            x0 = self._x
            pdf0 = self._pdf
            x1 = other._x
            pdf1 = other._pdf
            x = np.unique(np.hstack([x0, x1]))
            y0 = np.interp(x, x0, pdf0)
            y1 = np.interp(x, x1, pdf1)

            name = '({:s}) - ({:s})'.format(self.name, other.name)

            return Distribution(x, y0 - y1, name=name)
        elif hasattr(other, '__call__'):
            x0 = self._x
            y0 = self._pdf
            y1 = other(x0)

            n1 = getattr(other, '__name__', 'f(...)')
            if n1 == '<lambda>':
                t = inspect.getsource(other).replace(' ', '')[:-1]
                t = ''.join(t.split('lambda')[1:]).split(':')
                n1 = 'f({t[0]}) = {t[1]}'.format(t=t)
            name = '({:s}) - ({:s})'.format(self.name, n1)
            return Distribution(x0, y0 - y1, name=name)

    def __mul__(self, other):
        """ multiply distribution """
        if np.isscalar(other):
            name = '{1} * {0:s}'.format(self.name, other)
            return Distribution(self._x, self._pdf * other, name=name)
        elif isinstance(other, Distribution):
            x0 = self._x
            pdf0 = self._pdf
            x1 = other._x
            pdf1 = other._pdf
            x = np.unique(np.hstack([x0, x1]))
            y0 = np.interp(x, x0, pdf0)
            y1 = np.interp(x, x1, pdf1)

            name = '({:s}) * ({:s})'.format(self.name, other.name)
            return Distribution(x, y0 * y1, name=name)
        elif hasattr(other, '__call__'):
            x0 = self._x
            y0 = self._pdf
            y1 = other(x0)

            n1 = getattr(other, '__name__', 'f(...)')
            if n1 == '<lambda>':
                t = inspect.getsource(other).replace(' ', '')[:-1]
                t = ''.join(t.split('lambda')[1:]).split(':')
                n1 = 'f({t[0]}) = {t[1]}'.format(t=t)
            name = '({:s}) * ({:s})'.format(self.name, n1)
            return Distribution(x0, y0 * y1, name=name)

    def __div__(self, other):
        """ multiply distribution """
        if np.isscalar(other):
            name = '{:s} / {}'.format(self.name, other)
            return Distribution(self._x, self._pdf / other, name=name)
        elif isinstance(other, Distribution):
            x0 = self._x
            pdf0 = self._pdf
            x1 = other._x
            pdf1 = other._pdf
            x = np.unique(np.hstack([x0, x1]))
            y0 = np.interp(x, x0, pdf0)
            y1 = np.interp(x, x1, pdf1)

            name = '({:s}) / ({:s})'.format(self.name, other.name)
            return Distribution(x, y0 / y1, name=name)
        elif hasattr(other, '__call__'):
            x0 = self._x
            y0 = self._pdf
            y1 = other(x0)

            n1 = getattr(other, '__name__', 'f(...)')
            if n1 == '<lambda>':
                t = inspect.getsource(other).replace(' ', '')[:-1]
                t = ''.join(t.split('lambda')[1:]).split(':')
                n1 = 'f({t[0]}) = {t[1]}'.format(t=t)
            name = '({:s}) / ({:s})'.format(self.name, n1)
            return Distribution(x0, y0 / y1, name=name)

    def __repr__(self):
        return '{}\n{:s}'.format(object.__repr__(self), self.name)

    def __str__(self):
        return '{:s}'.format(self.name)

    def __call__(self, x):
        return self.pdf(x)


def main():
    """ Test case: combining 4 experimental measurements

        Let's have 4 independent measurements of the same quantity with Gaussian uncertainties.
        The measurements are samples of a given Gaussian distribution of which
        we want to estimate the mean and dispersion values

        A quick Bayesian inference (with uniform priors) will show that if all
        measurements are from the same distribution, the production of the 4
        posterior distributions will give you the underlying Gaussian
        parameters.

        if mu = {mk}, and sig = {sk} for k=1, N:
            p(m, s | mu, sig) ~ prod_k p(mk, sk | m, s) p(m, s)

        We also find that the product of Gaussians is a Gaussian
    """
    import pylab as plt

    #define a (normalized) gaussian probability distribution function
    Normal = lambda x, m, s: 1. / np.sqrt(2. * np.pi * s ** 2) * np.exp(-0.5 * ((x - m) / s) ** 2 )

    x = np.arange(0, 6, 0.01)
    mu = np.array([ 3.3,  2.65,  2.4,  3.14])
    sig = np.array([ 0.38,  0.17,  0.3,  0.34])
    yk = [Distribution(x, Normal(x, mk, sk), name='N({:0.3f},{:0.3f}'.format(mk, sk) ) for (mk, sk) in zip(mu, sig)]

    B = yk[0]
    for k in yk[1:]:
        B *= k

    print '{:6s} {:6s} {:6s}'.format(*'norm mean std'.split())
    for k in yk:
        print '{y.norm:5.3g} {y.mean:5.3g} {y.std:5.3g}'.format(y=k)
        plt.plot(x, k._pdf)

    plt.plot(x, B._pdf, lw=2, color='0.0')

    print "final distribution:"
    print "Expr: {B.name}\n stats: \n   mean = {B.mean},\n   std = {B.std},\n   skew = {B.skew},\n   kurtosis = {B.kurtosis}".format(B=B)


if __name__ == '__main__':
    main()
