import os
import random
import lmdb

def import_raw_data_samples(HYPER):

    """ """
    
    lmdb_env = lmdb.open(HYPER.PATH_TO_DATA_RAW_OPENCATALYST_IS2RE_TRAIN)
    
    print(lmdb_env.stat())
    return lmdb_env

