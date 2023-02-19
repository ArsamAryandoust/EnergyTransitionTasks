import os
import yaml
import random


def get_config_from_yaml:
    """
    Get config from yaml file
    """
    
    with open("config.yml", "r") as configfile:
        config = yaml.safe_load(configfile)

    return config


def check_create_dir(path):
    """
    Check if passed path exist and create if it doesn't.
    """
    
    if not os.path.isdir(path):
        os.mkdir(path)
        

def config_BE(config):
    """
    Augments configuration filefor processing Building Electricity dataset.
    """
    
    # get base config
    dictionary = config['building_electricity']
    
    # add data paths
    dictionary['path_to_raw_building_electricity'] = (
        config['general']['path_to_data_raw'] 
        + 'BuildingElectricity/profiles_400/'
    )
    dictionary['path_to_raw_building_year_profiles_file'] = (
        dictionary['path_to_raw_building_electricity']
        + 'building-year profiles/feature_scaled/2014 building-year profiles.csv'
    )
    dictionary['path_to_raw_meteo_data_folder'] = (
        dictionary['path_to_raw_building_electricity']
        + 'meteo data/'
    )
    dictionary['path_to_raw_aerial_imagery_file'] = (
        dictionary['path_to_raw_building_electricity']
        + 'building imagery/histogram/rgb/pixel_values.csv'
    )
    dictionary['path_to_data_building_electricity'] = (
        config['general']['path_to_data']
        + 'BuildingElectricity/'
    )
    dictionary['path_to_data_building_electricity_add'] = (
        dictionary['path_to_data_building_electricity']
        + 'additional/'
    )
    dictionary['path_to_data_building_electricity_train'] = (
        dictionary['path_to_data_building_electricity']
        + 'training/'
    )
    dictionary['path_to_data_building_electricity_val'] = (
        dictionary['path_to_data_building_electricity']
        + 'validation/'
    )
    dictionary['path_to_data_building_electricity_test'] = (
        dictionary['path_to_data_building_electricity']
        + 'testing/'
    )
    
    # create directory structure for saving results
    for path in [
        dictionary['path_to_data_building_electricity'], 
        dictionary['path_to_data_building_electricity_add'],
        dictionary['path_to_data_building_electricity_train'],
        dictionary['path_to_data_building_electricity_val'],
        dictionary['path_to_data_building_electricity_test']
    ]:
        check_create_dir(path)
        
    config['building_electricity'] = dictionary
    return config
    
    
def config_UM(config):
    """
    Augments configuration file for processing Uber Movement dataset.
    """
    
    # get base config
    dictionary = config['uber_movement']    
    
    # add data paths
    dictionary['path_to_data_raw_ubermovement'] = (
        config['general']['path_to_data_raw'] 
        + 'UberMovement/'
    )
    dictionary['path_to_data_ubermovement'] = (
        config['general']['path_to_data'] 
        + 'UberMovement/'
    )
    dictionary['path_to_data_ubermovement_add'] = (
        dictionary['path_to_data_ubermovement']
        + 'additional/'
    )
    dictionary['path_to_data_ubermovement_train'] = (
        dictionary['path_to_data_ubermovement']
        + 'training/'
    )
    dictionary['path_to_data_ubermovement_val'] = (
        dictionary['path_to_data_ubermovement']
        + 'validation/'
    )
    dictionary['path_to_data_ubermovement_test'] = (
        dictionary['path_to_data_ubermovement']
        + 'testing/'
    )
    
    # out of distribution test splitting rules in time
    random.seed(config['general']['seed'])
    quarter_of_year = random.sample(range(1,5), 1)
    random.seed(config['general']['seed'])
    hours_of_day = random.sample(range(24), 4)
    
    # dictionary saving rules
    dictionary['test_split_dict_ubermovement'] = {
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
    
    # Do some processing
    year_list = list(range(2015, 2021))
    quarter_list = ['-1-', '-2-', '-3-', '-4-']
    dictionary['ubermovement_list_of_cities'] = os.listdir(
        dictionary['path_to_data_raw_ubermovement']
    )
    dictionary['ubermovement_city_files_mapping'] = {}
    dictionary['ubermovement_city_id_mapping'] = {}
    for city_id, city in enumerate(dictionary['ubermovement_list_of_cities']):
        path_to_city = dictionary['path_to_data_raw_ubermovement'] + city + '/'
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
        dictionary['ubermovement_city_files_mapping'][city] = file_dict
        dictionary['ubermovement_city_id_mapping'][city] = city_id
  
    # create directory structure for saving results
    for path in [
        dictionary['path_to_data_raw_ubermovement'], 
        dictionary['path_to_data_raw_ubermovement_add'],
        dictionary['path_to_data_raw_ubermovement_train'],
        dictionary['path_to_data_raw_ubermovement_val'],
        dictionary['path_to_data_raw_ubermovement_test']
    ]:
        check_create_dir(path)
    
    config['uber_movement'] = dictionary
    return config
    
    
def config_CA(config):
    """
    Augments configuration file for processing ClimArt dataset.
    """
    pass      
    
    
def config_OC(config):
    """
    Augments configuration file for processing Open Catalyst dataset.
    """
    pass        
    
    
    
    
    
    

