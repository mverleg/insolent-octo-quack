
"""
	Ugly code, only need to run once anyway.
"""

from numpy import save
from os.path import join
from settings import BASE_DIR
from pandas import read_csv


# data = []
# with open(join(BASE_DIR, "data", "test.csv"), 'r') as fh:
# 	lines = fh.read().splitlines()
# 	print '{0:d} lines'.format(len(lines))
# 	for line in lines[1:]:
# 		data.append(line.split(','))
# data = zip(data)
# print data[0]


train = read_csv(join(BASE_DIR, "data", "train.csv"))
data = train.as_matrix()
save(join(BASE_DIR, 'train_raw_incl_id_cls.npy'), data)


test = read_csv(join(BASE_DIR, "data", "test.csv"))
data = test.as_matrix()
save(join(BASE_DIR, 'test_raw_incl_id_cls.npy'), data)


