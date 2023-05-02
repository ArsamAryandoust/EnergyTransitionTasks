import os
import math
import random
import shutil

import pandas as pd

 
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
      
      # create path
      check_create_dir(path)
      
  config_wind['subtask'] = subtask
  config_wind['seed'] = config['general']['seed']
  return config_wind
    
    
    
def check_create_dir(path: str):
  """
  Check if passed path exist and create if it doesn't.
  """
  if not os.path.isdir(path):
      os.mkdir(path)   
    
    
    
    
    
    
    
    
    

