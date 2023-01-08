from hyper import HyperParameter
import os
import random


class HyperOpenCatalyst(HyperParameter):

    """
    Bundles a bunch of hyper parameters.
    """
    
    
    
    def __init__(self):
    
        """ Set some paths by reading folders """
        
        # data paths
        self.PATH_TO_DATA_RAW_OPENCATALYST = HyperParameter.PATH_TO_DATA_RAW + 'OpenCatalyst/OC20/'
        self.PATH_TO_DATA_RAW_OPENCATALYST_PTE = HyperParameter.PATH_TO_DATA_RAW + 'OpenCatalyst/PubChemElements_all.csv'
        self.PATH_TO_DATA_RAW_OPENCATALYST_S2EF = self.PATH_TO_DATA_RAW_OPENCATALYST + 'S2EF/'
        self.PATH_TO_DATA_RAW_OPENCATALYST_S2EF_TRAIN = (
            self.PATH_TO_DATA_RAW_OPENCATALYST_S2EF 
            + 's2ef_train_2M/s2ef_train_2M/'
        )
        self.PATH_TO_DATA_RAW_OPENCATALYST_S2EF_VAL_ID = (
            self.PATH_TO_DATA_RAW_OPENCATALYST_S2EF 
            + 's2ef_val_id/s2ef_val_id/'
        )
        self.PATH_TO_DATA_RAW_OPENCATALYST_S2EF_VAL_OOD_ADS = (
            self.PATH_TO_DATA_RAW_OPENCATALYST_S2EF 
            + 's2ef_val_ood_ads/s2ef_val_ood_ads/'
        )
        self.PATH_TO_DATA_RAW_OPENCATALYST_S2EF_VAL_OOD_CAT = (
            self.PATH_TO_DATA_RAW_OPENCATALYST_S2EF 
            + 's2ef_val_ood_cat/s2ef_val_ood_cat/'
        )
        self.PATH_TO_DATA_RAW_OPENCATALYST_S2EF_VAL_OOD_BOTH = (
            self.PATH_TO_DATA_RAW_OPENCATALYST_S2EF 
            + 's2ef_val_ood_both/s2ef_val_ood_both/'
        )
        self.PATH_TO_DATA_OPENCATALYST = HyperParameter.PATH_TO_DATA + 'OpenCatalyst/'
        self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF = (
            self.PATH_TO_DATA_OPENCATALYST
            + 'OC20_S2EF/'
        )
        self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_ADD = (
            self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF 
            + 'additional/'
        )
        self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TRAIN = (
            self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF
            + 'training/'
        )
        self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_VAL = (
            self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF
            + 'validation/'
        )
        self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TEST = (
            self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF
            + 'testing/'
        )
        
        # Chunk size of data points per .csv file
        self.CHUNK_SIZE_OPENCATALYST = 150_000
        
        # Subsample values
        self.SUBSAMPLE_OPENCATALYST = 0.2
        
        ### Create directories for OpenCatalyst ###
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_OPENCATALYST)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_ADD)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TRAIN)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_VAL)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TEST)

