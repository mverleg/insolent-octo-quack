
"""
	Data transformations (e.g. symmetry corrections, histogram equalization, ...)

	Please note the unit tests and demo code in test/ and demo/ respectively.
"""

from numpy import cumsum, bincount


def equalize(X, mx = 255):
	"""
		Equalize the histogram of a one-dimensional integer vector X.

		:param X: Vector to equalize.
		:param mx: The maximum after equalization.

		Float division + rounding is not used in favor of integer division for speed.
	"""
	Xmin, Xmax = X.min(), X.max()
	assert Xmin - Xmax < 10000, 'spread in X is too large to equalize'
	Y = X - Xmin
	mp = cumsum(bincount(Y))
	mpmin = mp.min()
	Y = (mp[Y] - mpmin) * mx // (X.shape[0] - mpmin)
	return Y


