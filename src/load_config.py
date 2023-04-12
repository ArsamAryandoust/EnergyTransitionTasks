import os
import random
import math
import random
import shutil

import pandas as pd


        

def config_BE(config: dict, subtask: str, save: bool) -> dict:
  """
  Augments configuration filefor processing Building Electricity dataset.
  """
  # get base config
  config_building = config['BuildingElectricity']
  
  # fix little mismtach in subtask naming between raw data and desired results
  if subtask == 'buildings_92':
    raw_subtask = 'profiles_100'
  elif subtask == 'buildings_451':
    raw_subtask = 'profiles_400'
    
  # add raw data paths
  config_building['path_to_raw'] = (
    config['general']['path_to_data_raw'] 
    + 'BuildingElectricity/{}/'.format(raw_subtask))
  config_building['path_to_raw_building_year_profiles_file'] = (
    config_building['path_to_raw']
    + 'building-year profiles/feature_scaled/2014 building-year profiles.csv')
  config_building['path_to_raw_meteo_data_folder'] = (
    config_building['path_to_raw'] + 'meteo data/')
  config_building['path_to_raw_aerial_imagery_file'] = (
    config_building['path_to_raw']
    + 'building imagery/histogram/rgb/pixel_values.csv')
  
  # add results data paths
  config_building['path_to_data'] = (
    config['general']['path_to_data'] + 'BuildingElectricity/')
  config_building['path_to_data_subtask'] = (
    config_building['path_to_data'] + '{}/'.format(subtask))
  config_building['path_to_data_add'] = (
    config_building['path_to_data_subtask'] + 'additional/')
  config_building['path_to_data_train'] = (
    config_building['path_to_data_subtask'] + 'training/')
  config_building['path_to_data_val'] = (
    config_building['path_to_data_subtask'] + 'validation/')
  config_building['path_to_data_test'] = (
    config_building['path_to_data_subtask'] + 'testing/')
  
  # out of distribution test splitting rules in time
  random.seed(config['general']['seed'])
  ood_months = random.sample(range(1,13), 
    math.floor(12 * config_building['temporal_test_split']))
  random.seed(config['general']['seed'])
  ood_days = random.sample(range(1, 32),
    math.floor(31 * config_building['temporal_test_split']))
  random.seed(config['general']['seed'])
  ood_hours = random.sample(range(24),
    math.floor(24 * config_building['temporal_test_split']))
  random.seed(config['general']['seed'])
  ood_quarter_hours = random.sample([0, 15, 30, 45],
    math.floor(4 * config_building['temporal_test_split']))
  
  # out of distribution test splitting rules in space
  if subtask == 'buildings_92':
    n_buildings = 92
  elif subtask == 'buildings_451':
    n_buildings = 459 # ids go from 1-459, missing IDs hence 451 buildings
  random.seed(config['general']['seed'])
  ood_building_ids = random.sample(range(1, n_buildings+1), 
    math.floor(n_buildings * config_building['spatial_test_split']))
  
  # dictionary saving rules
  config_building['temporal_ood'] = {'ood_months': ood_months,
    'ood_days': ood_days, 'ood_hours': ood_hours,
    'ood_quarter_hours': ood_quarter_hours}
  config_building['spatial_ood'] = {'ood_building_ids': ood_building_ids}

  # if chosen to save create directory structure for saving results
  if save:
  
    if subtask == 'buildings_92' and os.path.isdir(
      config_building['path_to_data']):
      shutil.rmtree(config_building['path_to_data'])
    
    for path in [config_building['path_to_data'],
      config_building['path_to_data_subtask'], 
      config_building['path_to_data_add'],
      config_building['path_to_data_train'], 
      config_building['path_to_data_val'],
      config_building['path_to_data_test']]:
      check_create_dir(path)
      
  config_building['subtask'] = subtask
  config_building['seed'] = config['general']['seed']
  return config_building
  
    
def config_WF(config: dict, subtask: str, save: bool) -> dict:
  """
  Augments configuration file for processing Wind Farm dataset.
  """
  # get base config
  config_wind = config['WindFarm']
  
  # add data paths
  config_wind['path_to_data_raw'] = (config['general']['path_to_data_raw'] 
    + 'WindFarm/')
  config_wind['path_to_turb_loc_file'] = (config_wind['path_to_data_raw'] 
    + 'sdwpf_baidukddcup2022_turb_location.CSV')
  if subtask == 'days_245':
    config_wind['path_to_data_raw_file'] = (config_wind['path_to_data_raw']
      + 'wtbdata_245days.csv')
  elif subtask == 'days_177':
    config_wind['path_to_data_raw_infile_folder'] = (
      config_wind['path_to_data_raw'] + 'final_phase_test/infile/')
    config_wind['path_to_data_raw_outfile_folder'] = (
      config_wind['path_to_data_raw'] + 'final_phase_test/outfile/')
  config_wind['path_to_data'] = (config['general']['path_to_data'] 
    + 'WindFarm/')
  config_wind['path_to_data_subtask'] = (config_wind['path_to_data']
    + '{}/'.format(subtask))
  config_wind['path_to_data_train'] = (config_wind['path_to_data_subtask']
    + 'training/')
  config_wind['path_to_data_val'] = (config_wind['path_to_data_subtask']
    + 'validation/')
  config_wind['path_to_data_test'] = (config_wind['path_to_data_subtask']
    + 'testing/')
      
  ### out of distribution test splitting rules in time ###
  # days_245 has 245 days in total, days_177 has 177 in total
  if subtask== 'days_245':
    n_days = 245
  elif subtask=='days_177':
    n_days = 177
    
  # sample start days of blocks of block_size, here 14 days
  block_size = 14
  random.seed(config['general']['seed'])
  day_start_list = random.sample(range(1, n_days, block_size), 
    math.ceil(n_days * config_wind['temporal_test_split']/block_size))
    
  # extend the day list by entire block that is sampled
  ood_days = []
  for start_day in day_start_list:
    for day in range(start_day, start_day+block_size):
      ood_days.append(day)
  random.seed(config['general']['seed'])
  ood_hours = random.sample(range(24), 
    math.floor(24 * config_wind['temporal_test_split']))
  random.seed(config['general']['seed'])
  ood_minutes = random.sample(range(0, 60, 10), 
    math.floor(6 * config_wind['temporal_test_split']))
  
  ### out of distribution test splitting rules in space ###
  n_turbines = 134
  random.seed(config['general']['seed'])
  ood_turbine_ids = random.sample(range(1, n_turbines), 
    math.floor(n_turbines * config_wind['spatial_test_split']))
  
  # testing dictionaries
  config_wind['temporal_ood'] = {'ood_days': ood_days, 'ood_hours': ood_hours,
    'ood_minutes': ood_minutes}
  config_wind['spatial_ood'] = {'ood_turbine_ids': ood_turbine_ids}
  
  # create directory structure for saving results
  if save:
    
    if subtask == 'days_245' and os.path.isdir(config_wind['path_to_data']):
      shutil.rmtree(config_wind['path_to_data'])
    
    for path in [config_wind['path_to_data'],
      config_wind['path_to_data_subtask'], config_wind['path_to_data_train'], 
      config_wind['path_to_data_val'], config_wind['path_to_data_test']]:
      check_create_dir(path)
      
  config_wind['subtask'] = subtask
  config_wind['seed'] = config['general']['seed']
  return config_wind
    
    
def config_UM(config: dict, subtask: str, save=False) -> dict:
    """
    Augments configuration file for processing Uber Movement dataset.
    """
    # get base config
    config_uber = config['uber_movement']    
    
    # add data paths
    config_uber['path_to_data_raw'] = (config['general']['path_to_data_raw'] 
        + 'UberMovement/')
    config_uber['path_to_data'] = (config['general']['path_to_data'] 
        + 'UberMovement/')
    config_uber['path_to_data_subtask'] = (config_uber['path_to_data']
        + '{}/'.format(subtask))
    config_uber['path_to_data_add'] = (config_uber['path_to_data_subtask']
        + 'additional/')
    config_uber['path_to_data_train'] = (config_uber['path_to_data_subtask']
        + 'training/')
    config_uber['path_to_data_val'] = (config_uber['path_to_data_subtask']
        + 'validation/')
    config_uber['path_to_data_test'] = (config_uber['path_to_data_subtask']
        + 'testing/')
    
    # create list of citites and save to configuration dictionary
    random.seed(config['general']['seed'])
    list_of_cities = os.listdir(config_uber['path_to_data_raw'])
    random.shuffle(list_of_cities)
    if subtask == 'cities_10':
        list_of_cities = list_of_cities[:10]
    elif subtask == 'cities_20':
        list_of_cities = list_of_cities[:20]
    
    # out of distribution test splitting rules in time
    random.seed(config['general']['seed'])
    ood_years = random.sample(range(2015,2021), 
        math.floor(5 * config_uber['temporal_test_split']))
    ood_quarters_of_year = random.sample(range(1,5), 
        math.floor(4 * config_uber['temporal_test_split']))
    random.seed(config['general']['seed'])
    ood_hours = random.sample(range(24), 
        math.floor(24 * config_uber['temporal_test_split']))
    
    # out of distribution test splitting rules in space
    n_cities_test = round(config_uber['spatial_test_split'] * len(list_of_cities))
    random.seed(config['general']['seed'])
    ood_cities = random.sample(list_of_cities, n_cities_test)
    
    # dictionary saving rules
    config_uber['temporal_ood'] = {
        'ood_years': ood_years,
        'ood_quarters_of_year': ood_quarters_of_year,
        'ood_hours': ood_hours}
    config_uber['spatial_ood'] = {
            'ood_cities': ood_cities}
    
    # Create city files mapping and city id mapping
    year_list = list(range(2015, 2021))
    quarter_list = ['-1-', '-2-', '-3-', '-4-']
    config_uber['json'] = {}
    config_uber['csv_file_dict_list'] = {}
    config_uber['city_id_mapping'] = {}
    for city_id, city in enumerate(list_of_cities):
        path_to_city = config_uber['path_to_data_raw'] + city + '/'
        file_list = os.listdir(path_to_city)
        csv_file_dict_list = []
        for filename in file_list:
            if filename.endswith('.json'):
                json = filename
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
                
        # save 
        config_uber['json'][city] = json
        config_uber['csv_file_dict_list'][city] = csv_file_dict_list
        config_uber['city_id_mapping'][city] = city_id
    
    # create directory structure for saving results
    if save:
    
      if subtask == 'cities_10':
          config_uber['list_of_cities'] = list_of_cities[:10]
          
          if os.path.isdir(config_uber['path_to_data']):
              shutil.rmtree(config_uber['path_to_data'])
              
          for path in [config_uber['path_to_data'],
              config_uber['path_to_data_subtask'], 
              config_uber['path_to_data_add'],
              config_uber['path_to_data_train'], 
              config_uber['path_to_data_val'],
              config_uber['path_to_data_test']]:
              check_create_dir(path)
              
      elif subtask == 'cities_20':
          config_uber['list_of_cities'] = list_of_cities[10:20]
          # set full path to directory we want to copy
          path_to_copy_directory = config_uber['path_to_data'] + 'cities_10/'
          # copy directory into current subtask
          shutil.copytree(path_to_copy_directory, 
            config_uber['path_to_data_subtask'])
          
      elif subtask == 'cities_43':
          config_uber['list_of_cities'] = list_of_cities[20:]
          # set full path to directory we want to copy
          path_to_copy_directory = config_uber['path_to_data'] + 'cities_20/'
          # copy directory into current subtask
          shutil.copytree(path_to_copy_directory, 
            config_uber['path_to_data_subtask'])
        
      # create dataframe from dictionary
      df = pd.DataFrame.from_dict(config_uber['city_id_mapping'], 
        orient='index', columns=['city_id'])
      
      # save file
      saving_path = config_uber['path_to_data_add'] + 'city_to_id_mapping.csv'
      df.to_csv(saving_path)
    
    config_uber['subtask'] = subtask
    config_uber['seed'] = config['general']['seed']
    return config_uber
    

def config_CA(config: dict, subtask: str) -> dict:
    """
    Augments configuration file for processing ClimArt dataset.
    """
    # get base config
    config_climart = config['climart'] 
    
    # add data paths
    config_climart['path_to_data_raw'] = (
        config['general']['path_to_data_raw'] + 'ClimART/')
    config_climart['path_to_data_raw_inputs'] = (
        config_climart['path_to_data_raw'] + 'inputs/')
    config_climart['path_to_data_raw_outputs_subtask'] = (
        config_climart['path_to_data_raw'] + 'outputs_{}/'.format(subtask))
    config_climart['path_to_data'] = (
        config['general']['path_to_data'] + 'ClimART/')
    config_climart['path_to_data_subtask'] = (
        config_climart['path_to_data']+ '{}/'.format(subtask))
    config_climart['path_to_data_subtask_train'] = (
        config_climart['path_to_data_subtask'] + 'training/')
    config_climart['path_to_data_subtask_val'] = (
        config_climart['path_to_data_subtask'] + 'validation/')
    config_climart['path_to_data_subtask_test'] = (
        config_climart['path_to_data_subtask'] + 'testing/')
    
    # out of distribution test splitting rules in time
    year_list_test = [1850, 1851, 1852, 1991, 2097, 2098, 2099]
    t_step_size_h = 205
    n_t_steps_per_year = round(365 * 24 / t_step_size_h)
    hours_of_year_list = list(range(0, n_t_steps_per_year*t_step_size_h, 
        t_step_size_h))
    share_hours_sampling = 0.2
    n_hours_subsample = round(
        n_t_steps_per_year * config_climart['temporal_test_split'])
    random.seed(config['general']['seed'])
    hours_of_year_test = random.sample(hours_of_year_list, n_hours_subsample)
    
    # out of distribution test splitting rules in space
    n_lat, n_lon = 64, 128
    n_coordinates = n_lat * n_lon
    first_coordinates_index_list = list(range(n_coordinates))
    n_cord_subsample = round(
        config_climart['spatial_test_split'] * n_coordinates)
    random.seed(config['general']['seed'])
    coordinates_index_list = random.sample(first_coordinates_index_list,
        n_cord_subsample)
    
    coordinate_list = []
    for step in range(n_t_steps_per_year):
        coordinate_list_step = []
        for entry in coordinates_index_list:
            new_entry = entry + step * n_coordinates
            coordinate_list_step.append(new_entry)
            
        coordinate_list += coordinate_list_step
       
    # dictionary saving rules
    config_climart['temporal_ood'] = {
            'year': year_list_test,
            'hours_of_year': hours_of_year_test},
    config_climart['spatial_ood'] = {
            'coordinates': coordinate_list}
    
    # create directory structure for saving results
    if subtask == 'pristine':
        config_climart['datapoints_per_file'] = (
            config_climart['datapoints_per_file_pristine'])
        if os.path.isdir(config_climart['path_to_data']):
            shutil.rmtree(config_climart['path_to_data'])
    else:
        config_climart['datapoints_per_file'] = (
            config_climart['datapoints_per_file_pristine'])
            
    for path in [config_climart['path_to_data'], 
        config_climart['path_to_data_subtask'],
        config_climart['path_to_data_subtask_train'],
        config_climart['path_to_data_subtask_val'],
        config_climart['path_to_data_subtask_test']]:
        check_create_dir(path)
    
    config_climart['subtask'] = subtask
    config_climart['seed'] = config['general']['seed']
    return config_climart
    
    
    
    
    
    
    
    
def check_create_dir(path: str):
  """
  Check if passed path exist and create if it doesn't.
  """
  if not os.path.isdir(path):
      os.mkdir(path)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    
    
    
    

