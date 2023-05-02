import os
import random
import math
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
      
      # create path
      check_create_dir(path)
      
  config_building['subtask'] = subtask
  config_building['seed'] = config['general']['seed']
  return config_building
  
    
    
def check_create_dir(path: str):
  """
  Check if passed path exist and create if it doesn't.
  """
  if not os.path.isdir(path):
      os.mkdir(path)   
