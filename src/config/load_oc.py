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
  
  ###
  # Set paths to raw data files ###
  ###
  
  ### Set overarching paths to raw data ###
  config_opencat['path_to_data_raw'] = (
    config['general']['path_to_data_raw'] + 'OpenCatalyst/')
  config_opencat['path_to_data_raw_OC20'] = (
    config_opencat['path_to_data_raw'] + 'OC20/')
  config_opencat['path_to_data_raw_OC22'] = (
    config_opencat['path_to_data_raw'] + 'OC22/')
  
  ### Set paths to IS2RE/S for OC20 ###
  config_opencat['path_to_data_raw_OC20_is2res'] = (
    config_opencat['path_to_data_raw_OC20'] + 'is2re/all/')
  config_opencat['path_to_data_raw_OC20_is2res_train'] = (
    config_opencat['path_to_data_raw_OC20_is2res'] + 'train/')
  config_opencat['path_to_data_raw_OC20_is2res_val_id'] = (
    config_opencat['path_to_data_raw_OC20_is2res'] + 'val_id/')
  config_opencat['path_to_data_raw_OC20_is2res_val_ood_ads'] = (
    config_opencat['path_to_data_raw_OC20_is2res'] + 'val_ood_ads/')  
  config_opencat['path_to_data_raw_OC20_is2res_val_ood_cat'] = (
    config_opencat['path_to_data_raw_OC20_is2res'] + 'val_ood_cat/')  
  config_opencat['path_to_data_raw_OC20_is2res_val_ood_both'] = (
    config_opencat['path_to_data_raw_OC20_is2res'] + 'val_ood_both/') 
  config_opencat['path_to_data_raw_OC20_is2res_test_id'] = (
    config_opencat['path_to_data_raw_OC20_is2res'] + 'test_id/')
  config_opencat['path_to_data_raw_OC20_is2res_test_ood_ads'] = (
    config_opencat['path_to_data_raw_OC20_is2res'] + 'test_ood_ads/')  
  config_opencat['path_to_data_raw_OC20_is2res_test_ood_cat'] = (
    config_opencat['path_to_data_raw_OC20_is2res'] + 'test_ood_cat/')  
  config_opencat['path_to_data_raw_OC20_is2res_test_ood_both'] = (
    config_opencat['path_to_data_raw_OC20_is2res'] + 'test_ood_both/') 
  
  # test challenge data 2021
  config_opencat['path_to_data_raw_OC20_is2res_test_challenge'] = (
    config_opencat['path_to_data_raw_OC20'] + 'is2re_challenge/')  
    
  
  ### Set paths to S2EF for OC20 ###
  config_opencat['path_to_data_raw_OC20_s2ef'] = (
    config_opencat['path_to_data_raw_OC20'] + 's2ef/all/')
  config_opencat['path_to_data_raw_OC20_s2ef_train'] = (
    config_opencat['path_to_data_raw_OC20_s2ef'] + 'train/')
  config_opencat['path_to_data_raw_OC20_s2ef_val_id'] = (
    config_opencat['path_to_data_raw_OC20_s2ef'] + 'val_id/')
  config_opencat['path_to_data_raw_OC20_s2ef_val_ood_ads'] = (
    config_opencat['path_to_data_raw_OC20_s2ef'] + 'val_ood_ads/')  
  config_opencat['path_to_data_raw_OC20_s2ef_val_ood_cat'] = (
    config_opencat['path_to_data_raw_OC20_s2ef'] + 'val_ood_cat/')  
  config_opencat['path_to_data_raw_OC20_s2ef_val_ood_both'] = (
    config_opencat['path_to_data_raw_OC20_s2ef'] + 'val_ood_both/') 
  config_opencat['path_to_data_raw_OC20_s2ef_test_id'] = (
    config_opencat['path_to_data_raw_OC20_s2ef'] + 'test_id/')
  config_opencat['path_to_data_raw_OC20_s2ef_test_ood_ads'] = (
    config_opencat['path_to_data_raw_OC20_s2ef'] + 'test_ood_ads/')  
  config_opencat['path_to_data_raw_OC20_s2ef_test_ood_cat'] = (
    config_opencat['path_to_data_raw_OC20_s2ef'] + 'test_ood_cat/')  
  config_opencat['path_to_data_raw_OC20_s2ef_test_ood_both'] = (
    config_opencat['path_to_data_raw_OC20_s2ef'] + 'test_ood_both/') 
    
  # these have not been implemented yet. Data is still compressed
  config_opencat['path_to_data_raw_OC20_s2ef_md'] = (
    config_opencat['path_to_data_raw_OC20'] + 's2ef_md/')
  config_opencat['path_to_data_raw_OC20_s2ef_rattled'] = (
    config_opencat['path_to_data_raw_OC20'] + 's2ef_rattled/')
    
  
  ### Set paths to IS2RE/S for OC22 ###
  config_opencat['path_to_data_raw_OC22_is2res'] = (
    config_opencat['path_to_data_raw_OC22'] + 
    'is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/')
  config_opencat['path_to_data_raw_OC22_is2res_train'] = (
    config_opencat['path_to_data_raw_OC22_is2res'] + 'train/')
  config_opencat['path_to_data_raw_OC22_is2res_val_id'] = (
    config_opencat['path_to_data_raw_OC22_is2res'] + 'val_id/')
  config_opencat['path_to_data_raw_OC22_is2res_val_ood'] = (
    config_opencat['path_to_data_raw_OC22_is2res'] + 'val_ood/')
  config_opencat['path_to_data_raw_OC22_is2res_test_id'] = (
    config_opencat['path_to_data_raw_OC22_is2res'] + 'test_id/')
  config_opencat['path_to_data_raw_OC22_is2res_test_ood'] = (
    config_opencat['path_to_data_raw_OC22_is2res'] + 'test_ood/')    
  
  
  ### Set paths to S2EF for OC22 ###
  config_opencat['path_to_data_raw_OC22_s2ef'] = (
    config_opencat['path_to_data_raw_OC22'] + 
    's2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/')
  config_opencat['path_to_data_raw_OC22_s2ef_train'] = (
    config_opencat['path_to_data_raw_OC22_s2ef'] + 'train/')
  config_opencat['path_to_data_raw_OC22_s2ef_val_id'] = (
    config_opencat['path_to_data_raw_OC22_s2ef'] + 'val_id/')
  config_opencat['path_to_data_raw_OC22_s2ef_val_ood'] = (
    config_opencat['path_to_data_raw_OC22_s2ef'] + 'val_ood/')
  config_opencat['path_to_data_raw_OC22_s2ef_test_id'] = (
    config_opencat['path_to_data_raw_OC22_s2ef'] + 'test_id/')
  config_opencat['path_to_data_raw_OC22_s2ef_test_ood'] = (
    config_opencat['path_to_data_raw_OC22_s2ef'] + 'test_ood/')   
  
  
  ### Set paths to miscellaneous meta data ###
  
  # set path to Bader charge data for OC20
  config_opencat['path_to_data_raw_OC20_bader'] = (
    config_opencat['path_to_data_raw_OC20'] + 'oc20_bader_data/bader/')  
      
  # meta data for IS2RE/S test challenge in OC20
  config_opencat['path_to_data_raw_OC20_is2res_test_challenge_meta'] = (
    config_opencat['path_to_data_raw_OC20_is2res_test_challenge'] + 
    'metadata.npz')    
  
  # meta data for OC20
  config_opencat['path_to_data_raw_OC20_meta1'] = (
    config_opencat['path_to_data_raw_OC20'] + 'oc20_data_mapping.pkl')
  config_opencat['path_to_data_raw_OC20_meta2'] = (
    config_opencat['path_to_data_raw_OC20'] + 'mapping_adslab_slab.pkl')
  
  # meta data for OC22
  config_opencat['path_to_data_raw_OC22_meta1'] = (
    config_opencat['path_to_data_raw_OC22'] + 'oc22_metadata.pkl')
  config_opencat['path_to_data_raw_OC22_meta2'] = (
    config_opencat['path_to_data_raw_OC22'] + 'oc20_ref.pkl')
  
  
  ### Set paths to results ###
  
  config_opencat['path_to_data'] = (
    config['general']['path_to_data'] + 'OpenCatalyst/')
  config_opencat['path_to_data_subtask'] = (
    config_opencat['path_to_data'] + '{}/'.format(subtask))
  config_opencat['path_to_data_subtask_add'] = (
    config_opencat['path_to_data_subtask'] + 'additional/')
  config_opencat['path_to_data_subtask_train'] = (
    config_opencat['path_to_data_subtask'] + 'training/')
  config_opencat['path_to_data_subtask_val'] = (
    config_opencat['path_to_data_subtask'] + 'validation/')
  config_opencat['path_to_data_subtask_test'] = (
    config_opencat['path_to_data_subtask'] + 'testing/')
  
  
  
  ### Create saving folder structure ###
  if save:
    
    # create directory if not existent for entire task    
    check_create_dir(config_opencat['path_to_data'])
    
    # delete any previous results for this subtask
    if os.path.isdir(config_opencat['path_to_data_subtask']):
      shutil.rmtree(config_opencat['path_to_data_subtask'])
    
    # iterate over all directories
    for path in [config_opencat['path_to_data_subtask'],
      config_opencat['path_to_data_subtask_train'],
      config_opencat['path_to_data_subtask_val'],
      config_opencat['path_to_data_subtask_test'],
      config_opencat['path_to_data_subtask_add']]:
      
      # create directory if not existent      
      check_create_dir(path)
          
    
  
  
  
  
  
  
  
  ### Save some general values ###
  config_opencat['seed'] = config['general']['seed']
  config_opencat['subtask'] = subtask
  
  
  
  return config_opencat
    

    
def check_create_dir(path: str):
  """
  Check if passed path exist and create if it doesn't.
  """
  if not os.path.isdir(path):
      os.mkdir(path)   
    

