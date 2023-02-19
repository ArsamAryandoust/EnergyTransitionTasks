import os
import yaml

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
    
    # data path
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
        
    
def config_UM(config):
    """
    Augments configuration file for processing Uber Movement dataset.
    """
    pass    
    
    
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
    
    
    
    
    
    

