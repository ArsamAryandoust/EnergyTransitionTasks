import os
import random
import math
import shutil

import pandas as pd



def config_CA(config: dict, subtask: str, save: bool) -> dict:
  """
  Augments configuration file for processing ClimArt dataset.
  """
  # get base config
  config_climart = config['ClimART'] 
  
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
    'hours_of_year': hours_of_year_test
  }
  config_climart['spatial_ood'] = {
    'coordinates': coordinate_list
  }
  
  # create directory structure for saving results
  
  if subtask == 'pristine':
    config_climart['data_per_file'] = config_climart['data_per_file_pristine']
        
    if save:
    
      if os.path.isdir(config_climart['path_to_data']):
        shutil.rmtree(config_climart['path_to_data'])
          
  elif subtask == 'clear_sky':
    config_climart['data_per_file'] = config_climart['data_per_file_clearsky']
   
  if save:   
    
    # iterate over all directories
    for path in [config_climart['path_to_data'], 
      config_climart['path_to_data_subtask'],
      config_climart['path_to_data_subtask_train'],
      config_climart['path_to_data_subtask_val'],
      config_climart['path_to_data_subtask_test']]:
      # create directory if not existent      
      check_create_dir(path)
  
  # set subtask and return
  config_climart['subtask'] = subtask
  config_climart['seed'] = config['general']['seed']
  
  
  return config_climart
  
    
def check_create_dir(path: str):
  """
  Check if passed path exist and create if it doesn't.
  """
  if not os.path.isdir(path):
      os.mkdir(path)   
    
    
    
    
    

