
from os.path import join
from numpy import array, save, load
from settings import BASE_DIR
from re import compile


fname = 'train'

rgx = compile(r'''((?:[^,"']|"[^"]*"|'[^']*')+)''')
data = []
length = None

with open(join(BASE_DIR, 'data', '{0:s}.csv'.format(fname)), 'r') as fh:
	""" Line-by-line so as to not consume much memory """
	linenr = 0
	for line in fh:
		linenr += 1
		""" Use regex in case commas are escaped; http://stackoverflow.com/a/2787064/723090 """
		parts = rgx.split(line.rstrip('\n'))[1::2]
		if length is None:
			length = len(parts)
			for k in range(length):
				data.append([])
		if not length == len(parts):
			print 'line {0:d} has length {1:d} instead of known length {2:d}; it will be skipped'.format(linenr, len(parts), length)
			continue
		for indx, value in enumerate(parts):
			data[indx].append(value)
		if not linenr % 1000:
			print 'reading line {0:d}'.format(linenr)

for colnr, column in enumerate(data):
	arr = array(column[1:])
	save(join(BASE_DIR, 'data', '{0:s}_row{1:03d}.npy'.format(fname, colnr)), arr)
	if not colnr % 25:
		print 'saving column {0:d}'.format(colnr)


#for nr, column in enumerate(data):
#	print load(join(BASE_DIR, 'data', 'row{0:03d}.npy'.format(nr)))


