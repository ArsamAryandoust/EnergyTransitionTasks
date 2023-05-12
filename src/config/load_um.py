import os
import random
import math
import random
import shutil

import pandas as pd


        
    
def config_UM(config: dict, subtask: str, save=False) -> dict:
  """
  Augments configuration file for processing Uber Movement dataset.
  """
  # get base config
  config_uber = config['UberMovement']    
  
  # add data paths
  config_uber['path_to_data_raw'] = (config['general']['path_to_data_raw'] 
    + 'UberMovement/')
  config_uber['path_to_data'] = (config['general']['path_to_data'] 
    + 'UberMovement/')
  config_uber['path_to_data_meta'] = (config_uber['path_to_data']
    + 'metadata/')
  config_uber['path_to_data_subtask'] = (config_uber['path_to_data']
    + '{}/'.format(subtask))
  config_uber['path_to_data_train'] = (config_uber['path_to_data_subtask']
    + 'training/')
  config_uber['path_to_data_val'] = (config_uber['path_to_data_subtask']
    + 'validation/')
  config_uber['path_to_data_test'] = (config_uber['path_to_data_subtask']
    + 'testing/')
  
  # create list of citites and save to configuration dictionary
  random.seed(config['general']['seed'])
  list_of_cities = list_of_cities = os.listdir(config_uber['path_to_data_raw'])
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
    'ood_hours': ood_hours
  }
  config_uber['spatial_ood'] = {
    'ood_cities': ood_cities
  }
  
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
  if subtask == 'cities_10':
    list_of_cities = list_of_cities[:10]
  elif subtask == 'cities_20':
    list_of_cities = list_of_cities[10:20]
  elif subtask == 'cities_43':
    list_of_cities = list_of_cities[20:]
    
  if save:
  
    if subtask == 'cities_10':
      
      if os.path.isdir(config_uber['path_to_data']):
        shutil.rmtree(config_uber['path_to_data'])
          
      for path in [config_uber['path_to_data'],
        config_uber['path_to_data_subtask'], 
        config_uber['path_to_data_meta'],
        config_uber['path_to_data_train'], 
        config_uber['path_to_data_val'],
        config_uber['path_to_data_test']]:
        
        # create path
        check_create_dir(path)
            
    elif subtask == 'cities_20':

      # set full path to directory we want to copy
      path_to_copy_directory = config_uber['path_to_data'] + 'cities_10/'
      
      # copy directory into current subtask
      shutil.copytree(path_to_copy_directory, 
        config_uber['path_to_data_subtask'])
      
    elif subtask == 'cities_43':
      # set full path to directory we want to copy
      path_to_copy_directory = config_uber['path_to_data'] + 'cities_20/'
      
      # copy directory into current subtask
      shutil.copytree(path_to_copy_directory, 
        config_uber['path_to_data_subtask'])
      
    # create dataframe from dictionary
    df = pd.DataFrame.from_dict(config_uber['city_id_mapping'], 
      orient='index', columns=['city_id'])
    
    # save file
    saving_path = config_uber['path_to_data_meta'] + 'id_mapping_{}.csv'.format(
      subtask)
    df.to_csv(saving_path)
  
  config_uber['list_of_cities'] = list_of_cities
  config_uber['subtask'] = subtask
  config_uber['seed'] = config['general']['seed']
  return config_uber
  
    
    
    
def check_create_dir(path: str):
  """
  Check if passed path exist and create if it doesn't.
  """
  if not os.path.isdir(path):
      os.mkdir(path)   
    
    
    
   
    
    
