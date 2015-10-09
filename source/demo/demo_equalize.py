from numpy import array, cumsum, histogram
from matplotlib.pyplot import subplots, show

from utils.transform import equalize


def plot_equalize():
	X = array(sum([range(110, 121)] * 5 + [range(130, 141)] * 7, []))
	Xc = cumsum(histogram(X, bins = 30)[0])
	Y = equalize(X)
	Yc = cumsum(histogram(Y, bins = 30)[0])
	fig, ax = subplots()
	ax.plot(Xc, c = 'b')
	ax.plot(Yc, c = 'r')
	show()


if __name__ == '__main__':
	plot_equalize()


