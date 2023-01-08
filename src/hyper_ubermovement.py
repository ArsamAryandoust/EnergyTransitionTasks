from hyper import HyperParameter
import os
import random


class HyperUberMovement(HyperParameter):

    """
    Bundles a bunch of hyper parameters.
    """
    
    
    def __init__(self):
    
        """ """
        
        # data paths
        self.PATH_TO_DATA_RAW_UBERMOVEMENT = HyperParameter.PATH_TO_DATA_RAW + 'UberMovement/'
        self.PATH_TO_DATA_UBERMOVEMENT = HyperParameter.PATH_TO_DATA + 'UberMovement/'
        self.PATH_TO_DATA_UBERMOVEMENT_ADDITIONAL = self.PATH_TO_DATA_UBERMOVEMENT + 'additional/'
        self.PATH_TO_DATA_UBERMOVEMENT_TRAIN = self.PATH_TO_DATA_UBERMOVEMENT + 'training/'
        self.PATH_TO_DATA_UBERMOVEMENT_VAL = self.PATH_TO_DATA_UBERMOVEMENT + 'validation/'
        self.PATH_TO_DATA_UBERMOVEMENT_TEST = self.PATH_TO_DATA_UBERMOVEMENT + 'testing/'

        # Chunk size of data points per .csv file
        self.CHUNK_SIZE_UBERMOVEMENT = 20_000_000
       
        # share to split training and validation data
        self.TRAIN_VAL_SPLIT_UBERMOVEMENT = 0.8
        
        # Subsample values
        self.SUBSAMPLE_UBERMOVEMENT = 0.01
        
        # out of distribution test splitting rules in time
        random.seed(HyperParameter.SEED)
        quarter_of_year = random.sample(range(1,5), 1)
        random.seed(HyperParameter.SEED)
        hours_of_day = random.sample(range(24), 4)
        
        # dictionary saving rules
        self.TEST_SPLIT_DICT_UBERMOVEMENT = {
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
        
        
        
        ### Uber Movement ###
        year_list = list(range(2015, 2021))
        quarter_list = ['-1-', '-2-', '-3-', '-4-']
        self.UBERMOVEMENT_LIST_OF_CITIES = os.listdir(self.PATH_TO_DATA_RAW_UBERMOVEMENT)
        self.UBERMOVEMENT_LIST_OF_CITIES.remove('Sydney') # Sydney is problematic for current applications, remove it
        self.UBERMOVEMENT_LIST_OF_CITIES.remove('Perth') # Perth is problematic for current applications, remove it
        self.UBERMOVEMENT_LIST_OF_CITIES = self.UBERMOVEMENT_LIST_OF_CITIES[:14] # shorten city list to 14 ideally
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
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_UBERMOVEMENT)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_UBERMOVEMENT_ADDITIONAL)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_UBERMOVEMENT_TEST)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_UBERMOVEMENT_VAL)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_UBERMOVEMENT_TRAIN)
        
