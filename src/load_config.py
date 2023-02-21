import os
import yaml
import random
import math

def get_config_from_yaml() -> dict:
    """
    Get config from yaml file
    """
    
    with open("config.yml", "r") as configfile:
        config = yaml.safe_load(configfile)

    return config


def check_create_dir(path: str):
    """
    Check if passed path exist and create if it doesn't.
    """
    
    if not os.path.isdir(path):
        os.mkdir(path)
        

def config_BE(config: dict, subtask: str) -> dict:
    """
    Augments configuration filefor processing Building Electricity dataset.
    """
    
    # get base config
    dictionary = config['building_electricity']
    
    # add data paths
    dictionary['path_to_raw'] = (
        config['general']['path_to_data_raw'] 
        + 'BuildingElectricity/{}/'.format(subtask)
    )
    dictionary['path_to_raw_building_year_profiles_file'] = (
        dictionary['path_to_raw']
        + 'building-year profiles/feature_scaled/2014 building-year profiles.csv'
    )
    dictionary['path_to_raw_meteo_data_folder'] = (
        dictionary['path_to_raw']
        + 'meteo data/'
    )
    dictionary['path_to_raw_aerial_imagery_file'] = (
        dictionary['path_to_raw']
        + 'building imagery/histogram/rgb/pixel_values.csv'
    )
    dictionary['path_to_data'] = (
        config['general']['path_to_data']
        + 'BuildingElectricity/'
    )
    dictionary['path_to_data_subtask'] = (
        dictionary['path_to_data']
        + '{}/'.format(subtask)
    )
    dictionary['path_to_data_add'] = (
        dictionary['path_to_data_subtask']
        + 'additional/'
    )
    dictionary['path_to_data_train'] = (
        dictionary['path_to_data_subtask']
        + 'training/'
    )
    dictionary['path_to_data_val'] = (
        dictionary['path_to_data_subtask']
        + 'validation/'
    )
    dictionary['path_to_data_test'] = (
        dictionary['path_to_data_subtask']
        + 'testing/'
    )
    
    # out of distribution test splitting rules in time
    random.seed(config['general']['seed'])
    month_list = random.sample(
        range(1,13), 
        math.floor(12 * dictionary['temporal_test_split'])
    )
    random.seed(config['general']['seed'])
    day_list = random.sample(
        range(1, 32),
        math.floor(31 * dictionary['temporal_test_split'])        
    )
    random.seed(config['general']['seed'])
    hour_list = random.sample(
        range(24),
        math.floor(24 * dictionary['temporal_test_split'])        
    )
    random.seed(config['general']['seed'])
    quarter_hour_list = random.sample(
        [0, 15, 30, 45],
        math.floor(4 * dictionary['temporal_test_split'])       
    )
    
    # out of distribution test splitting rules in space
    if subtask == 'building_92':
        n_buildings = 92
    elif subtask == 'building_451':
        n_buildings = 459 # ids go from 1-459, missing IDs hence 451 buildings
        
    random.seed(config['general']['seed'])
    building_id_list = random.sample(
        range(1, n_buildings+1), 
        math.floor(n_buildings * dictionary['spatial_test_split'])
    )
    
    # dictionary saving rules
    dictionary['ood_split_dict'] = {
        'temporal_dict': {
            'month_list': month_list,
            'day_list': day_list,
            'hour_list': hour_list,
            'quarter_hour_list': quarter_hour_list
        },
        'spatial_dict': {
            'building_id_list': building_id_list
        }
    }
    
    # create directory structure for saving results
    for path in [
        dictionary['path_to_data'],
        dictionary['path_to_data_subtask'],
        dictionary['path_to_data_add'],
        dictionary['path_to_data_train'],
        dictionary['path_to_data_val'],
        dictionary['path_to_data_test']
    ]:
        check_create_dir(path)
        
    config['building_electricity'] = dictionary
    return config
    
    
def config_UM(config: dict) -> dict:
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
    
    
def config_CA(config: dict) -> dict:
    """
    Augments configuration file for processing ClimArt dataset.
    """
    
    # get base config
    dictionary = config['climart']
    
    # add data paths
    dictionary['path_to_data_raw_climart'] = (
        config['general']['path_to_data_raw'] 
        + 'ClimART/'
    )
    dictionary['path_to_raw_climart_statistics'] = (
        dictionary['path_to_data_raw_climart']
        + 'statistics/'
    )
    dictionary['path_to_raw_climart_inputs'] = (
        dictionary['path_to_data_raw_climart']
        + 'inputs/'
    )
    dictionary['path_to_raw_climart_outputs_clear_sky'] = (
        dictionary['path_to_data_raw_climart']
        + 'outputs_clear_sky/'
    )
    dictionary['path_to_raw_climart_outputs_pristine'] = (
        dictionary['path_to_data_raw_climart']
        + 'outputs_pristine/'
    )
    dictionary['path_to_data_climart'] = (
        config['general']['path_to_data'] 
        + 'ClimART/'
    )
    dictionary['path_to_data_climart_clearsky'] = (
        dictionary['path_to_data_climart']
        + 'clear_sky/'
    )
    dictionary['path_to_data_climart_pristine'] = (
        dictionary['path_to_data_climart']
        + 'pristine/'
    )
    dictionary['path_to_data_climart_clearsky_train'] = (
        dictionary['path_to_data_climart_clearsky']
        + 'training/'
    )
    dictionary['path_to_data_climart_pristine_train'] = (
        dictionary['path_to_data_climart_pristine']
        + 'training/'
    )
    dictionary['path_to_data_climart_clearsky_val'] = (
        dictionary['path_to_data_climart_clearsky']
        + 'validation/'
    )
    dictionary['path_to_data_climart_pristine_val'] = (
        dictionary['path_to_data_climart_pristine']
        + 'validation/'
    )
    dictionary['path_to_data_climart_clearsky_test'] = (
        dictionary['path_to_data_climart_clearsky']
        + 'testing/'
    )
    dictionary['path_to_data_climart_pristine_test'] = (
        dictionary['path_to_data_climart_pristine']
        + 'testing/'
    )
    
    
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
    dictionary['test_plit_dict_climart'] = {
        'temporal_dict': {
            'year': 2014,
            'hours_of_year': hours_of_year
        },
        'spatial_dict': {
            'coordinates': coordinate_list
        }
    }
    
    # save a list with names of meta file names
    dictionary['climart_meta_filenames_dict'] = {
        'meta':'META_INFO.json',
        'stats':'statistics.npz'
    }
    
    # create directory structure for saving results
    for path in [
        dictionary['path_to_data_climart'], 
        dictionary['path_to_data_climart_clearsky'],
        dictionary['path_to_data_climart_pristine'],
        dictionary['path_to_data_climart_clearsky_train'],
        dictionary['path_to_data_climart_pristine_train'],
        dictionary['path_to_data_climart_clearsky_val'],
        dictionary['path_to_data_climart_pristine_val'],
        dictionary['path_to_data_climart_clearsky_test'],
        dictionary['path_to_data_climart_pristine_test']
    ]:
        check_create_dir(path)
    
    config['climart'] = dictionary
    return config
    
    
def config_OC(config: dict) -> dict:
    """
    Augments configuration file for processing Open Catalyst dataset.
    """
    
    # get base config
    dictionary = config['open_catalyst']
            
    # add data paths
    dictionary['path_to_data_raw_opencatalyst'] = (
        config['general']['path_to_data_raw'] 
        + 'OpenCatalyst/OC20/'
    )
    dictionary['path_to_data_raw_opencatalyst_pte'] = (
        config['general']['path_to_data_raw'] 
        + 'OpenCatalyst/PubChemElements_all.csv'
    )
    dictionary['path_to_data_raw_opencatalyst_s2ef'] = (
        dictionary['path_to_data_raw_opencatalyst']
        + 'S2EF/'
    )
    dictionary['path_to_data_raw_opencatalyst_s2ef_train'] = (
        dictionary['path_to_data_raw_opencatalyst_s2ef']
        + 's2ef_train_2M/'
    )
    dictionary['path_to_data_raw_opencatalyst_s2ef_val_id'] = (
        dictionary['path_to_data_raw_opencatalyst_s2ef']
        + 's2ef_val_id/'
    )
    dictionary['path_to_data_raw_opencatalyst_s2ef_val_ood_ads'] = (
        dictionary['path_to_data_raw_opencatalyst_s2ef']
        + 's2ef_val_ood_ads/'
    )
    dictionary['path_to_data_raw_opencatalyst_s2ef_val_ood_cat'] = (
        dictionary['path_to_data_raw_opencatalyst_s2ef']
        + 's2ef_val_ood_cat/'
    )
    dictionary['path_to_data_raw_opencatalyst_s2ef_val_ood_both'] = (
        dictionary['path_to_data_raw_opencatalyst_s2ef']
        + 's2ef_val_ood_both/'
    )
    dictionary['path_to_data_opencatalyst'] = (
        config['general']['path_to_data'] 
        + 'OpenCatalyst/'
    )
    dictionary['path_to_data_opencatalyst_oc20_s2ef'] = (
        dictionary['path_to_data_opencatalyst']
        + 'OC20_S2EF/'
    )
    dictionary['path_to_data_opencatalyst_oc20_s2ef_add'] = (
        dictionary['path_to_data_opencatalyst_oc20_s2ef']
        + 'additional/'
    )
    dictionary['path_to_data_opencatalyst_oc20_s2ef_train'] = (
        dictionary['path_to_data_opencatalyst_oc20_s2ef']
        + 'training/'
    )
    dictionary['path_to_data_opencatalyst_oc20_s2ef_val'] = (
        dictionary['path_to_data_opencatalyst_oc20_s2ef']
        + 'validation/'
    )
    dictionary['path_to_data_opencatalyst_oc20_s2ef_test'] = (
        dictionary['path_to_data_opencatalyst_oc20_s2ef']
        + 'testing/'
    )
    
    # create directory structure for saving results
    for path in [
        dictionary['path_to_data_opencatalyst'], 
        dictionary['path_to_data_opencatalyst_oc20_s2ef'],
        dictionary['path_to_data_opencatalyst_oc20_s2ef_add'],
        dictionary['path_to_data_opencatalyst_oc20_s2ef_train'],
        dictionary['path_to_data_opencatalyst_oc20_s2ef_val'],
        dictionary['path_to_data_opencatalyst_oc20_s2ef_test']
    ]:
        check_create_dir(path)
    
    
    config['open_catalyst'] = dictionary
    return config
    
    
    
    

