from hyper import HyperParameter
import os
import random
import math


class HyperClimart(HyperParameter):

    """
    Bundles a bunch of hyper parameters.
    """
    
    
    def __init__(self):
    
        """ Set some paths by reading folders """
        
        # data paths
        self.PATH_TO_DATA_RAW_CLIMART = HyperParameter.PATH_TO_DATA_RAW + 'ClimART/'
        self.PATH_TO_DATA_RAW_CLIMART_STATISTICS = self.PATH_TO_DATA_RAW_CLIMART + 'statistics/'
        self.PATH_TO_DATA_RAW_CLIMART_INPUTS = self.PATH_TO_DATA_RAW_CLIMART + 'inputs/'
        self.PATH_TO_DATA_RAW_CLIMART_OUTPUTS_CLEAR_SKY = self.PATH_TO_DATA_RAW_CLIMART + 'outputs_clear_sky/'
        self.PATH_TO_DATA_RAW_CLIMART_OUTPUTS_PRISTINE = self.PATH_TO_DATA_RAW_CLIMART + 'outputs_pristine/'
        self.PATH_TO_DATA_CLIMART = HyperParameter.PATH_TO_DATA + 'ClimART/'
        self.PATH_TO_DATA_CLIMART_CLEARSKY = self.PATH_TO_DATA_CLIMART + 'clear_sky/'
        self.PATH_TO_DATA_CLIMART_PRISTINE = self.PATH_TO_DATA_CLIMART + 'pristine/'
        self.PATH_TO_DATA_CLIMART_CLEARSKY_TRAIN = self.PATH_TO_DATA_CLIMART_CLEARSKY + 'training/'
        self.PATH_TO_DATA_CLIMART_PRISTINE_TRAIN = self.PATH_TO_DATA_CLIMART_PRISTINE + 'training/'
        self.PATH_TO_DATA_CLIMART_CLEARSKY_VAL = self.PATH_TO_DATA_CLIMART_CLEARSKY + 'validation/'
        self.PATH_TO_DATA_CLIMART_PRISTINE_VAL = self.PATH_TO_DATA_CLIMART_PRISTINE + 'validation/'
        self.PATH_TO_DATA_CLIMART_CLEARSKY_TEST = self.PATH_TO_DATA_CLIMART_CLEARSKY + 'testing/'
        self.PATH_TO_DATA_CLIMART_PRISTINE_TEST = self.PATH_TO_DATA_CLIMART_PRISTINE + 'testing/'
        
        # Chunk size of data points per .csv file
        self.CHUNK_SIZE_CLIMART = 35_000
        
        # share to split training and validation data
        self.TRAIN_VAL_SPLIT_CLIMART = 0.8
        
        # Subsample share
        self.SUBSAMPLE_CLIMART = 0.2
        
        # out of distribution test splitting rules in time
        t_step_size_h = 205
        n_t_steps_per_year = math.ceil(365 * 24 / t_step_size_h)
        hours_of_year_list = list(range(0, n_t_steps_per_year*t_step_size_h, t_step_size_h))
        share_hours_sampling = 0.2
        n_hours_subsample = math.ceil(n_t_steps_per_year * share_hours_sampling)
        random.seed(HyperParameter.SEED)
        hours_of_year = random.sample(
            hours_of_year_list, 
            n_hours_subsample
        )
        
        # out of distribution test splitting rules in space
        n_lat, n_lon = 64, 128
        n_coordinates = n_lat * n_lon
        first_coordinates_index_list = list(range(n_coordinates))
        share_coordinates_sampling = 0.2
        n_cord_subsample = math.ceil(share_coordinates_sampling * n_coordinates)
        random.seed(HyperParameter.SEED)
        coordinates_index_list = random.sample(
            first_coordinates_index_list,
            n_cord_subsample
        )
        
        coordinate_list = []
        for step in range(n_t_steps_per_year):
            
            coordinate_list_step = []
            for entry in coordinates_index_list:
                new_entry = entry + step * n_coordinates
                coordinate_list_step.append(new_entry)
                
            coordinate_list += coordinate_list_step
            
        # dictionary saving rules
        self.TEST_SPLIT_DICT_CLIMART = {
            'temporal_dict': {
                'year': 2014,
                'hours_of_year': hours_of_year
            },
            'spatial_dict': {
                'coordinates': coordinate_list
            }
        }
            
        
        # save a list with names of meta file names
        self.CLIMART_META_FILENAMES_DICT = {
            'meta':'META_INFO.json',
            'stats':'statistics.npz'
        }
        
        
        ### Create directories for CLimART ###
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_CLIMART)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_CLIMART_CLEARSKY)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_CLIMART_PRISTINE)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_CLIMART_CLEARSKY_TRAIN)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_CLIMART_PRISTINE_TRAIN)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_CLIMART_CLEARSKY_VAL)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_CLIMART_PRISTINE_VAL)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_CLIMART_CLEARSKY_TEST)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_CLIMART_PRISTINE_TEST)
        
