
from numpy import load
from os.path import join
from settings import BASE_DIR


data = load(join(BASE_DIR, 'data', 'train_raw_incl_id_cls.npy'))


