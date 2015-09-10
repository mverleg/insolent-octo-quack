
"""
	Loading and saving of data.
"""

from os.path import join
import gc
from tempfile import mktemp, gettempdir
from numpy import array, uint8, save, vstack, load, hstack
from numpy.core.defchararray import strip, replace
from os import remove
from settings import BASE_DIR
from re import compile


rgx = compile(r'''((?:[^,"']|"[^"]*"|'[^']*')+)''')
fpath_default = join(BASE_DIR, 'data', 'train.csv')


def load_data(fpath = fpath_default, row_limit = None, row_skip = None):
	with open(fpath, 'r') as fh:
		""" Line-by-line so as to not consume much memory """
		linenr, length, data, clss, ids = 0, None, [], [], []
		header = [h.strip('"') for h in rgx.split(fh.readline().rstrip('\n'))[1::2]]
		if row_skip:
			for k in range(row_skip):
				fh.readline()
			print('skipped {0:d} rows'.format(row_skip))
		for line in fh:
			linenr += 1
			if linenr % 500 == 0:
				print('reading line {0:d}...'.format(row_skip + linenr))
			if row_limit is not None and linenr > row_limit:
				break
			""" Use regex in case commas are escaped; http://stackoverflow.com/a/2787064/723090 """
			parts = rgx.split(line.rstrip('\n'))[1::2]
			ids.append(parts[0])
			clss.append(parts[-1])
			if length is None:
				length = len(parts[1:-1])
				for k in range(length):
					data.append([])
			if not length == len(parts) - 2:
				print('line {0:d} has length {1:d} instead of known length {2:d}; it will be skipped'.format(linenr, len(parts) - 2, length))
				continue
			for indx, value in enumerate(parts[1:-1]):
				data[indx].append(value)
	print('converting data')
	header = array(header[1:-1])
	data = array(data)
	labels = array(clss).astype(uint8)
	ids = array(ids).astype(uint8)
	print('data in "{0:s}" loaded!'.format(fpath))
	return header, data, labels, ids


def to_ints_only(data):
	conv = []
	failed = []
	colnr = 0
	for col in data:
		colnr += 1
		if colnr % 100 == 0:
			print 'converting column {0:d}...'.format(colnr)
		col = replace(col, '""', '0')
		col = replace(col, 'NA', '0')
		col = replace(col, 'false', '0')
		col = replace(col, 'true', '1')
		col = strip(col, '"')
		try:
			irow = col.astype(int)
		except ValueError as err:
			failed.append(str(err).split(':', 1)[1])
		else:
			conv.append(irow)
	print('failed for', failed)
	print('{0:d} columns removed'.format(len(failed)))
	return array(conv)


def batch_load_ints(fpath = fpath_default, batch_size = 8000):
	cursor = 0
	tmpdir = gettempdir()
	bfiles = []
	while True:
		header, data, labels, ids = load_data(fpath = fpath, row_limit = cursor + batch_size, row_skip = cursor)
		gc.collect()  # free memory
		idata = to_ints_only(data)
		bname = join(tmpdir, 'iobatch{0:d}.npy'.format(cursor))
		save(bname, idata)
		print('saved batch to "{0:s}"'.format(bname))
		bfiles.append(bname)
		cursor += batch_size
		row_count = data.shape[1]
		del data, idata
		gc.collect()
		if row_count < batch_size:
			print('stopping since last batch reached end of file')
			break
	parts = []
	for bpath in bfiles:
		parts.append(load(bpath))
		remove(bpath)
	merged = hstack(parts)
	del parts
	gc.collect()
	return merged


if __name__ == '__main__':
	data = batch_load_ints()
	print('final shape', data.shape)


