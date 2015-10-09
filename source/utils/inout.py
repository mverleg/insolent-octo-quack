
import gc
from os.path import join
from tempfile import gettempdir
from numpy import array, uint8, save, load, hstack, int16
from numpy.core.defchararray import strip, replace
from os import remove
from re import compile
from settings import BASE_DIR


rgx = compile(r'''((?:[^,"']|"[^"]*"|'[^']*')+)''')
fpath_default = join(BASE_DIR, 'data', 'train.csv')
skiprows = []
skip = (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 717, 718, 721, 722, 724, 725, 726, 728, 729, 730, 731, 732, 733, 734, 735, 736, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 756, 757, 758, 759, 760, 761, 762, 763, 764, 766, 767, 768, 769, 770, 771, 772, 773, 774, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 814, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 875, 876, 877, 878, 879, 880, 881, 882, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 1931)


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
			print('converting column {0:d}...'.format(colnr))
		col = strip(col, '"')
		col = replace(col, '', '0')
		col = replace(col, 'NA', '0')
		col = replace(col, 'false', '0')
		col = replace(col, 'true', '1')
		try:
			irow = col.astype(int16)
		except ValueError as err:
			skiprows.append(colnr - 1)
			failed.append(str(err).split(':', 1)[1])
		except OverflowError as err:
			print(str(err))
			skiprows.append(colnr - 1)
		# except OverflowError as err:
		# 	print(str(err))
		# 	print('will look for overflow error value...')
		# 	for v in col:
		# 		try:
		# 			v.astype(int)
		# 		except:
		# 			print 'overflow:', v
		else:
			conv.append(irow)
		del col
		gc.collect()  # free memory
	print('failed for (excluding overflows): "{0:s}"'.format('", "'.join(failed)))
	print('{0:d} columns removed'.format(len(failed)))
	return array(conv)


def remove_cols(data, skip_cols):
	conv = []
	colnr = 0
	for col in data:
		if colnr % 200 == 0:
			print('processing column {0:d}...'.format(colnr))
			gc.collect()
		if colnr not in skip_cols:
			col = strip(col, '"')
			col = replace(col, '', '0')
			col = replace(col, 'NA', '0')
			col = replace(col, 'false', '0')
			col = replace(col, 'true', '1')
			conv.append(col.astype(int16))
		colnr += 1
	gc.collect()
	return array(conv)


def batch_load_ints(fpath = fpath_default, batch_size = 10000):
	cursor = 0
	tmpdir = gettempdir()
	bfiles = []
	marge_labels = []
	while True:
		header, data, labels, ids = load_data(fpath = fpath, row_limit = batch_size, row_skip = cursor)
		marge_labels.append(labels)
		gc.collect()  # free memory
		#idata = to_ints_only(data)
		idata = remove_cols(data, skip_cols = skip)
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
	print('rows to skip:')
	print(set(skiprows))
	print('going to merge data')
	merged = load(bfiles[0])
	for bpath in bfiles[1:]:
		merged = hstack((merged, load(bpath),))
		remove(bpath)
	gc.collect()
	return merged.T, header, hstack(marge_labels)


if __name__ == '__main__':
	print('skipping {0:d} columns'.format(len(skip)))
	data, header, labels = batch_load_ints()
	print(labels.shape)
	save(join(BASE_DIR, 'data', 'merged.npy'), data)
	save(join(BASE_DIR, 'data', 'header.npy'), header)
	save(join(BASE_DIR, 'data', 'labels.npy'), labels)
	print('final shape', data.shape, 'type', data.dtype)


