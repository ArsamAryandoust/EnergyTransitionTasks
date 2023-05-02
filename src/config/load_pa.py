import os
import random
import math
import shutil

import pandas as pd


  
def config_PA(config: dict, subtask: str, save: bool) -> dict:
  """
  Augments configuration file for processing Polianna dataset.
  """
  # get base config
  config_polianna = config['Polianna'] 
  
  # add data paths
  config_polianna['path_to_data_raw'] = (
    config['general']['path_to_data_raw'] + 'Polianna/')
  config_polianna['path_to_data_raw_meta'] = (
    config_polianna['path_to_data_raw'] + '01_policy_info/')
  config_polianna['path_to_data_raw_dataframe'] = (
    config_polianna['path_to_data_raw'] + '02_processed_to_dataframe/')
  
  config_polianna['path_to_data'] = (
    config['general']['path_to_data'] + 'Polianna/')
  config_polianna['path_to_data_meta'] = (
    config_polianna['path_to_data'] + 'metadata/') 
  config_polianna['path_to_data_subtask'] = (
    config_polianna['path_to_data'] + '{}/'.format(subtask))
  config_polianna['path_to_data_subtask_add'] = (
    config_polianna['path_to_data_subtask'] + 'additional/')
  config_polianna['path_to_data_subtask_train'] = (
    config_polianna['path_to_data_subtask'] + 'training/')
  config_polianna['path_to_data_subtask_val'] = (
    config_polianna['path_to_data_subtask'] + 'validation/')
  config_polianna['path_to_data_subtask_test'] = (
    config_polianna['path_to_data_subtask'] + 'testing/')
  
  # out of distribution test splitting rules in time
 
  
  # out of distribution test splitting rules in space
 
 
     
  # dictionary saving rules
  config_polianna['temporal_ood'] = {
  }
  config_polianna['spatial_ood'] = {
  }
  
  # create directory structure for saving results
  if save:
    
    if subtask == 'article_level':
      if os.path.isdir(config_polianna['path_to_data']):
        shutil.rmtree(config_polianna['path_to_data'])
    
    # iterate over all directories
    for path in [config_polianna['path_to_data'],
      config_polianna['path_to_data_meta'],
      config_polianna['path_to_data_subtask'],
      config_polianna['path_to_data_subtask_add'],
      config_polianna['path_to_data_subtask_train'],
      config_polianna['path_to_data_subtask_val'],
      config_polianna['path_to_data_subtask_test']]:
      
      # create directory if not existent      
      check_create_dir(path)
  
  # set subtask and return
  config_polianna['subtask'] = subtask
  config_polianna['seed'] = config['general']['seed']
  
  
  return config_polianna    
    
    
    
    
def check_create_dir(path: str):
  """
  Check if passed path exist and create if it doesn't.
  """
  if not os.path.isdir(path):
      os.mkdir(path)   
    
    
    
   
    
    
    

