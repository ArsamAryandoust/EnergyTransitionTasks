import os
import random
import math
import shutil

import pandas as pd


        
    
def config_OC(config: dict, subtask: str, save: bool) -> dict:
  """
  Augments configuration file for processing Open Catalyst dataset.
  """
  # get base config
  config_opencat = config['OpenCatalyst']     
  
  
  
  config_opencat['seed'] = config['general']['seed']
  config_opencat['subtask'] = subtask
  
  
  return config_opencat
    
    
"""    
def config_OC(config: dict) -> dict:
    
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
"""    
    
def check_create_dir(path: str):
  """
  Check if passed path exist and create if it doesn't.
  """
  if not os.path.isdir(path):
      os.mkdir(path)   
    

