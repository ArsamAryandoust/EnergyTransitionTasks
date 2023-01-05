import os
import random
import math

class HyperParameter:

    """
    Bundles a bunch of hyper parameters.
    """
    
    # Choose which data to process
    PROCESS_OPENCATALYST = False
    PROCESS_UBERMOVEMENT = True
    PROCESS_CLIMART = False
    
    # Choose which data to shuffle
    SHUFFLE_UBERMOVEMENT = False
    SHUFFLE_CLIMART = False
    
    
    # Random seed
    SEED = 3
    
    ### Data paths ###
    
    # general
    PATH_TO_DATA = '../data/'
    PATH_TO_DATA_RAW = PATH_TO_DATA + 'raw/'
    
    # Uber Movement
    PATH_TO_DATA_RAW_UBERMOVEMENT = PATH_TO_DATA_RAW + 'UberMovement/'
    PATH_TO_DATA_UBERMOVEMENT = PATH_TO_DATA + 'UberMovement/'
    PATH_TO_DATA_UBERMOVEMENT_ADDITIONAL = PATH_TO_DATA_UBERMOVEMENT + 'additional/'
    PATH_TO_DATA_UBERMOVEMENT_TRAIN = PATH_TO_DATA_UBERMOVEMENT + 'training/'
    PATH_TO_DATA_UBERMOVEMENT_VAL = PATH_TO_DATA_UBERMOVEMENT + 'validation/'
    PATH_TO_DATA_UBERMOVEMENT_TEST = PATH_TO_DATA_UBERMOVEMENT + 'testing/'
    
    # ClimART
    PATH_TO_DATA_RAW_CLIMART = PATH_TO_DATA_RAW + 'ClimART/'
    PATH_TO_DATA_RAW_CLIMART_STATISTICS = PATH_TO_DATA_RAW_CLIMART + 'statistics/'
    PATH_TO_DATA_RAW_CLIMART_INPUTS = PATH_TO_DATA_RAW_CLIMART + 'inputs/'
    PATH_TO_DATA_RAW_CLIMART_OUTPUTS_CLEAR_SKY = PATH_TO_DATA_RAW_CLIMART + 'outputs_clear_sky/'
    PATH_TO_DATA_RAW_CLIMART_OUTPUTS_PRISTINE = PATH_TO_DATA_RAW_CLIMART + 'outputs_pristine/'
    PATH_TO_DATA_CLIMART = PATH_TO_DATA + 'ClimART/'
    PATH_TO_DATA_CLIMART_CLEARSKY = PATH_TO_DATA_CLIMART + 'clear_sky/'
    PATH_TO_DATA_CLIMART_PRISTINE = PATH_TO_DATA_CLIMART + 'pristine/'
    PATH_TO_DATA_CLIMART_CLEARSKY_TRAIN = PATH_TO_DATA_CLIMART_CLEARSKY + 'training/'
    PATH_TO_DATA_CLIMART_PRISTINE_TRAIN = PATH_TO_DATA_CLIMART_PRISTINE + 'training/'
    PATH_TO_DATA_CLIMART_CLEARSKY_VAL = PATH_TO_DATA_CLIMART_CLEARSKY + 'validation/'
    PATH_TO_DATA_CLIMART_PRISTINE_VAL = PATH_TO_DATA_CLIMART_PRISTINE + 'validation/'
    PATH_TO_DATA_CLIMART_CLEARSKY_TEST = PATH_TO_DATA_CLIMART_CLEARSKY + 'testing/'
    PATH_TO_DATA_CLIMART_PRISTINE_TEST = PATH_TO_DATA_CLIMART_PRISTINE + 'testing/'
    
    
    # Open Catalyst
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
    
    
    ### Training, validation, testing splits ###
    
    # Chunk size of data points per .csv file
    CHUNK_SIZE_UBERMOVEMENT = 20_000_000
    CHUNK_SIZE_CLIMART = 35_000
    CHUNK_SIZE_OPENCATALYST = 150_000
    
    # share to split training and validation data
    TRAIN_VAL_SPLIT_UBERMOVEMENT = 0.8
    TRAIN_VAL_SPLIT_CLIMART = 0.8
    
    # Subsample values
    SUBSAMPLE_CLIMART = 0.2
    SUBSAMPLE_OPENCATALYST = 0.1
    
    # out of distribution test splitting rules in time and space
    random.seed(SEED)
    quarter_of_year = random.sample(range(1,5), 1)
    random.seed(SEED)
    hours_of_day = random.sample(range(24), 4)
    
    TEST_SPLIT_DICT_UBERMOVEMENT = {
        'temporal_dict': {
            'year': 2017,
            'quarter_of_year': quarter_of_year,
            'hours_of_day': hours_of_day
        },
        'spatial_dict': {
            'city_share': 0.1,
            'city_zone_share': 0.1
        }
    }
    
    
    t_step_size_h = 205
    n_t_steps_per_year = math.ceil(365 * 24 / t_step_size_h)
    hours_of_year_list = list(range(0, n_t_steps_per_year*t_step_size_h, t_step_size_h))
    share_hours_sampling = 0.2
    n_hours_subsample = math.ceil(n_t_steps_per_year * share_hours_sampling)
    random.seed(SEED)
    hours_of_year = random.sample(
        hours_of_year_list, 
        n_hours_subsample
    )
    
    n_lat, n_lon = 64, 128
    n_coordinates = n_lat * n_lon
    first_coordinates_index_list = list(range(n_coordinates))
    share_coordinates_sampling = 0.2
    n_cord_subsample = math.ceil(share_coordinates_sampling * n_coordinates)
    random.seed(SEED)
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
        
    TEST_SPLIT_DICT_CLIMART = {
        'temporal_dict': {
            'year': 2014,
            'hours_of_year': hours_of_year
        },
        'spatial_dict': {
            'coordinates': coordinate_list
        }
    }
    
    
    ### Methods ###
    
    def __init__(self):
    
        """ Set some paths by reading folders """
        
        ### Uber Movement ###
        year_list = list(range(2015, 2021))
        quarter_list = ['-1-', '-2-', '-3-', '-4-']
        self.UBERMOVEMENT_LIST_OF_CITIES = os.listdir(self.PATH_TO_DATA_RAW_UBERMOVEMENT)
        self.UBERMOVEMENT_LIST_OF_CITIES = self.UBERMOVEMENT_LIST_OF_CITIES[:14] # shorten city list
        self.UBERMOVEMENT_CITY_FILES_MAPPING = {}
        self.UBERMOVEMENT_CITY_ID_MAPPING = {}
        for city_id, city in enumerate(self.UBERMOVEMENT_LIST_OF_CITIES):
            path_to_city = self.PATH_TO_DATA_RAW_UBERMOVEMENT + city + '/'
            file_list = os.listdir(path_to_city)
            csv_file_dict_list = []
            for filename in file_list:
                if filename.endswith('.json'):
                    json = filename
                    break
                    
                else:
                    # declare new empty directory to be filled with desired values
                    csv_file_dict = {}
                    
                    # determine if weekday
                    if 'OnlyWeekdays' in filename:
                        daytype = 1
                    elif 'OnlyWeekends' in filename:
                        daytype = 0
                    
                    # determine year
                    for year in year_list:
                        if str(year) in filename:
                            break
                            
                    # determine quarter of year
                    for quarter_of_year in quarter_list:
                        if quarter_of_year in filename:
                            quarter_of_year = int(quarter_of_year[1])
                            break
                    
                    # fill dictionary with desired values
                    csv_file_dict['daytype'] = daytype
                    csv_file_dict['year'] = year
                    csv_file_dict['quarter_of_year'] = quarter_of_year
                    csv_file_dict['filename'] = filename
                    
                    # append csv file dictionary to list
                    csv_file_dict_list.append(csv_file_dict)
                    
            # create file name dictionary
            file_dict = {
                'json' : json,
                'csv_file_dict_list': csv_file_dict_list
            }
            
            # save 
            self.UBERMOVEMENT_CITY_FILES_MAPPING[city] = file_dict
            self.UBERMOVEMENT_CITY_ID_MAPPING[city] = city_id
            
       
        ### Create directories for Uber Movement ###
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT)
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT_ADDITIONAL)
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT_TEST)
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT_VAL)
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT_TRAIN)
        
        
        ### ClimART ###
        
        # save a list with names of meta file names
        self.CLIMART_META_FILENAMES_DICT = {
            'meta':'META_INFO.json',
            'stats':'statistics.npz'
        }
        
        
        
        ### Create directories for CLimART ###
        self.check_create_dir(self.PATH_TO_DATA_CLIMART)
        self.check_create_dir(self.PATH_TO_DATA_CLIMART_CLEARSKY)
        self.check_create_dir(self.PATH_TO_DATA_CLIMART_PRISTINE)
        self.check_create_dir(self.PATH_TO_DATA_CLIMART_CLEARSKY_TRAIN)
        self.check_create_dir(self.PATH_TO_DATA_CLIMART_PRISTINE_TRAIN)
        self.check_create_dir(self.PATH_TO_DATA_CLIMART_CLEARSKY_VAL)
        self.check_create_dir(self.PATH_TO_DATA_CLIMART_PRISTINE_VAL)
        self.check_create_dir(self.PATH_TO_DATA_CLIMART_CLEARSKY_TEST)
        self.check_create_dir(self.PATH_TO_DATA_CLIMART_PRISTINE_TEST)
        
           
        ### Create directories for OpenCatalyst ###
        self.check_create_dir(self.PATH_TO_DATA_OPENCATALYST)
        self.check_create_dir(self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF)
        self.check_create_dir(self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_ADD)
        self.check_create_dir(self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TRAIN)
        self.check_create_dir(self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_VAL)
        self.check_create_dir(self.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TEST)
        

    def check_create_dir(self, path):
    
        """ """
        
        if not os.path.isdir(path):
            os.mkdir(path)
