from glob import glob
import json
from copy import copy
from os.path import join

from numpy import load, array
from matplotlib.pyplot import subplots, show, close

from settings import BASE_DIR
from utils.transform import equalize

basefname = 'train'


def istype(v, tp = int):
	try:
		tp(v)
	except ValueError:
		return False
	return True


confpath = join(BASE_DIR, 'member', 'mark', 'format.json')
format = json.load(open(confpath))  # todo


try:
	print 'press ctrl+C (and possibly enter) to stop'
	for colnr, fname in enumerate(glob('{0:s}*'.format(join(BASE_DIR, 'data', basefname)))[:20]):
		""" Load data """
		column = load(fname)
		""" Remove quotes """
		column = [v.strip('\'').strip('"') for v in column]
		""" Replace all the NA values before converting """
		columnnonnum = array([v for v in column if not istype(v, int)], dtype = object)
		if len(columnnonnum):
			print 'non-num for {0:d}:'.format(colnr), columnnonnum
			continue
		column = array([int(v) for v in column], dtype = int)
		fig, ((ax_frac, ax_), (ax_before, ax_after), (ax_before_cdf, ax_after_cdf)) = subplots(3, 2, figsize = (8, 10))
		fig.suptitle('column {0:d}'.format(colnr))
		fig.tight_layout()
		ax_before.hist(column, facecolor = 'orange', bins = 30)
		ax_before_cdf.hist(column, cumulative = True, facecolor = 'red', bins = 30)
		cut_cnt = 0
		if str(colnr) in format:
			fmt = format[str(colnr)]
		else:
			fmt = copy(format['default'])
		if 'shift' in fmt:
			print 'shifting by {0}'.format(fmt['shift'])
			column += fmt['shift']
		if 'cut_gt' in fmt and 'cut_to' in fmt:
			print 'cutting > {0:f} to {1:f} for {2:d}'.format(fmt['cut_gt'], fmt['cut_to'], colnr)
			cut_cnt += (column > fmt['cut_gt']).sum()
			column[column > fmt['cut_gt']] = fmt['cut_to']
		if 'cut_lt' in fmt and 'cut_to' in fmt:
			print 'cutting < {0:f} to {1:f} for {2:d}'.format(fmt['cut_lt'], fmt['cut_to'], colnr)
			cut_cnt += (column < fmt['cut_lt']).sum()
			column[column < fmt['cut_lt']] = fmt['cut_to']
		if 'equalize' in fmt and fmt['equalize']:
			print 'equalizing histogram'
			column = equalize(column)
		ax_after.hist(column, facecolor = 'blue', bins = 30)
		ax_after_cdf.hist(column, cumulative = True, facecolor = 'green', bins = 30)
		ax_frac.pie([column.shape[0] - columnnonnum.shape[0] - cut_cnt, columnnonnum.shape[0], cut_cnt], labels = ['normal', 'NaN', 'cut'])
		print 'frac', column.shape[0], [column.shape[0] - columnnonnum.shape[0] - cut_cnt, columnnonnum.shape[0], cut_cnt]
		show(block = False)
		while True:
			if 'cut_gt' in fmt:
				print 'column {0:d} already has a cutoff at {1:d}'.format(colnr, fmt['cut_gt'])
				close()
				break
			cut = raw_input('column {0:d} cutoff point? '.format(colnr))
			if cut.strip() == '':
				print 'no cutoff'
				break
			try:
				cut = int(cut)
			except:
				print 'invalid (non-int) input {0}',format(cut)
			else:
				print 'cut at', cut
				fmt['cut_gt'] = cut
				fmt['cut_to'] = 0
				format[str(colnr)] = fmt
				close()
				break
except KeyboardInterrupt:
	conf_json = json.dumps(format, indent = 2, sort_keys = True)
	print conf_json
	while True:
		answer = raw_input('save? [y/n] ').strip().lower()
		if answer == 'y':
			with open(confpath, 'w+') as fh:
				fh.write(conf_json)
			print 'saved to', confpath
			break
		if answer == 'n':
			break


