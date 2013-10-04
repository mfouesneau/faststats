FASTSTATS - fast algorithm to do statistics
===========================================


This package is my current exploration on how to make fast statistics on big
data.  Functions are typically **several orders of magnitude faster**, or so
they claim.


I recently discovered how slow certain algorithms in numpy/scipy could be very
robust but very slow because they have to handle many tests and dimensions and
multipurpose usage and so on. They are _most_ legit implementation decisions.
However when you deal with tons of data, say 10^7 points in a 10 or 20
dimensions, the slightest overhead could end up overloading your computer and
potentially crash your system.

**Note**: algorithms in this package as usage targeted. This is how we can speed
up the algorithms.

Quick example
-------------

```python
from scipy.stats import gaussian_kde

def npkde(x, xe):
   kde = gaussian_kde(x)
   r = kde.evaluate(xe)
   return r

x = np.random.normal(0, 1, 1e6)
xe = np.linspace(0., 1., 256)

%timeit fastkde1D(x)
10 loops, best of 3: 31.9 ms per loop

%timeit npkde(x, xe)
1 loops, best of 3: 11.8 s per loop
```

The result is a **~ 10 ^ 4 speed up** !!! Results are identical
Note that ``gaussian_kde`` is not optimized for this specific application. 
