
"""
	py.test style unit tests
"""

from numpy import array, linspace, histogram, cumsum, sqrt

from utils.transform import equalize


def test_equalize():
	"""
		Besides basic properties, does a linearity test (that may not be mathematically sound)
	"""
	X = array(sum([range(10, 21)] * 15 + [range(130, 141)] * 4, []))
	Y = equalize(X, 255)
	assert Y.min() == 0
	assert Y.max() == 255
	C = cumsum(histogram(Y, bins = 30)[0])
	L = linspace(0, X.shape[0], 30)
	err = sqrt(sum((L - C)**2)) / X.shape[0]
	assert err < 0.5, 'equalized cdf appears to be not be linear ({0:.2f} rmse from linearity)'.format(err)


