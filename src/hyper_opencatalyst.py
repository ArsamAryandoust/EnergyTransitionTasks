from hyper import HyperParamter
import os
import random
import math

class HyperOpenCatalyst(HyperParamter):

    """
    Bundles a bunch of hyper parameters.
    """
    
    # data paths
    PATH_TO_DATA_RAW_OPENCATALYST = PATH_TO_DATA_RAW + 'OpenCatalyst/OC20/'
    PATH_TO_DATA_RAW_OPENCATALYST_PTE = PATH_TO_DATA_RAW + 'OpenCatalyst/PubChemElements_all.csv'
    PATH_TO_DATA_RAW_OPENCATALYST_S2EF = PATH_TO_DATA_RAW_OPENCATALYST + 'S2EF/'
    PATH_TO_DATA_RAW_OPENCATALYST_S2EF_TRAIN = (
        PATH_TO_DATA_RAW_OPENCATALYST_S2EF 
        + 's2ef_train_2M/s2ef_train_2M/'
    )
    PATH_TO_DATA_RAW_OPENCATALYST_S2EF_VAL_ID = (
        PATH_TO_DATA_RAW_OPENCATALYST_S2EF 
        + 's2ef_val_id/s2ef_val_id/'
    )
    PATH_TO_DATA_RAW_OPENCATALYST_S2EF_VAL_OOD_ADS = (
        PATH_TO_DATA_RAW_OPENCATALYST_S2EF 
        + 's2ef_val_ood_ads/s2ef_val_ood_ads/'
    )
    PATH_TO_DATA_RAW_OPENCATALYST_S2EF_VAL_OOD_CAT = (
        PATH_TO_DATA_RAW_OPENCATALYST_S2EF 
        + 's2ef_val_ood_cat/s2ef_val_ood_cat/'
    )
    PATH_TO_DATA_RAW_OPENCATALYST_S2EF_VAL_OOD_BOTH = (
        PATH_TO_DATA_RAW_OPENCATALYST_S2EF 
        + 's2ef_val_ood_both/s2ef_val_ood_both/'
    )
    PATH_TO_DATA_OPENCATALYST = PATH_TO_DATA + 'OpenCatalyst/'
    PATH_TO_DATA_OPENCATALYST_OC20_S2EF = (
        PATH_TO_DATA_OPENCATALYST
        + 'OC20_S2EF/'
    )
    PATH_TO_DATA_OPENCATALYST_OC20_S2EF_ADD = (
        PATH_TO_DATA_OPENCATALYST_OC20_S2EF 
        + 'additional/'
    )
    PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TRAIN = (
        PATH_TO_DATA_OPENCATALYST_OC20_S2EF
        + 'training/'
    )
    PATH_TO_DATA_OPENCATALYST_OC20_S2EF_VAL = (
        PATH_TO_DATA_OPENCATALYST_OC20_S2EF
        + 'validation/'
    )
    PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TEST = (
        PATH_TO_DATA_OPENCATALYST_OC20_S2EF
        + 'testing/'
    )
    
    # Chunk size of data points per .csv file
    CHUNK_SIZE_OPENCATALYST = 150_000
    
    # Subsample values
    SUBSAMPLE_OPENCATALYST = 0.01
    
    
    def __init__(self):
    
        """ Set some paths by reading folders """
         
        ### Create directories for OpenCatalyst ###
        self.check_create_dir(self.PATH_TO_DATA_OPENCATALYST)
        self.check_create_dir(self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF)
        self.check_create_dir(self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_ADD)
        self.check_create_dir(self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TRAIN)
        self.check_create_dir(self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_VAL)
        self.check_create_dir(self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TEST)

