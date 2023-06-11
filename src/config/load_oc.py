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
  config_opencat['path_to_data_raw_oc20'] = (
    config_opencat['path_to_data_raw'] + 'OC20/')
  config_opencat['path_to_data_raw_oc22'] = (
    config_opencat['path_to_data_raw'] + 'OC22/')
  
  ### Set paths to IS2RE/S for oc20 ###
  config_opencat['path_to_data_raw_oc20_is2res'] = (
    config_opencat['path_to_data_raw_oc20'] + 'is2re/all/')
  config_opencat['path_to_data_raw_oc20_is2res_train'] = (
    config_opencat['path_to_data_raw_oc20_is2res'] + 'train/')
  config_opencat['path_to_data_raw_oc20_is2res_val_id'] = (
    config_opencat['path_to_data_raw_oc20_is2res'] + 'val_id/')
  config_opencat['path_to_data_raw_oc20_is2res_val_ood_ads'] = (
    config_opencat['path_to_data_raw_oc20_is2res'] + 'val_ood_ads/')  
  config_opencat['path_to_data_raw_oc20_is2res_val_ood_cat'] = (
    config_opencat['path_to_data_raw_oc20_is2res'] + 'val_ood_cat/')  
  config_opencat['path_to_data_raw_oc20_is2res_val_ood_both'] = (
    config_opencat['path_to_data_raw_oc20_is2res'] + 'val_ood_both/') 
  config_opencat['path_to_data_raw_oc20_is2res_test_id'] = (
    config_opencat['path_to_data_raw_oc20_is2res'] + 'test_id/')
  config_opencat['path_to_data_raw_oc20_is2res_test_ood_ads'] = (
    config_opencat['path_to_data_raw_oc20_is2res'] + 'test_ood_ads/')  
  config_opencat['path_to_data_raw_oc20_is2res_test_ood_cat'] = (
    config_opencat['path_to_data_raw_oc20_is2res'] + 'test_ood_cat/')  
  config_opencat['path_to_data_raw_oc20_is2res_test_ood_both'] = (
    config_opencat['path_to_data_raw_oc20_is2res'] + 'test_ood_both/') 
  
  # test challenge data 2021
  config_opencat['path_to_data_raw_oc20_is2res_test_challenge'] = (
    config_opencat['path_to_data_raw_oc20'] + 'is2re_test_challenge_2021/')  
    
  
  ### Set paths to S2EF for oc20 ###
  config_opencat['path_to_data_raw_oc20_s2ef'] = (
    config_opencat['path_to_data_raw_oc20'] + 's2ef/all/')
  config_opencat['path_to_data_raw_oc20_s2ef_train'] = (
    config_opencat['path_to_data_raw_oc20_s2ef'] + 'train/')
  config_opencat['path_to_data_raw_oc20_s2ef_val_id'] = (
    config_opencat['path_to_data_raw_oc20_s2ef'] + 'val_id/')
  config_opencat['path_to_data_raw_oc20_s2ef_val_ood_ads'] = (
    config_opencat['path_to_data_raw_oc20_s2ef'] + 'val_ood_ads/')  
  config_opencat['path_to_data_raw_oc20_s2ef_val_ood_cat'] = (
    config_opencat['path_to_data_raw_oc20_s2ef'] + 'val_ood_cat/')  
  config_opencat['path_to_data_raw_oc20_s2ef_val_ood_both'] = (
    config_opencat['path_to_data_raw_oc20_s2ef'] + 'val_ood_both/') 
  config_opencat['path_to_data_raw_oc20_s2ef_test_id'] = (
    config_opencat['path_to_data_raw_oc20_s2ef'] + 'test_id/')
  config_opencat['path_to_data_raw_oc20_s2ef_test_ood_ads'] = (
    config_opencat['path_to_data_raw_oc20_s2ef'] + 'test_ood_ads/')  
  config_opencat['path_to_data_raw_oc20_s2ef_test_ood_cat'] = (
    config_opencat['path_to_data_raw_oc20_s2ef'] + 'test_ood_cat/')  
  config_opencat['path_to_data_raw_oc20_s2ef_test_ood_both'] = (
    config_opencat['path_to_data_raw_oc20_s2ef'] + 'test_ood_both/') 
    
  # these have not been implemented yet. Data is still compressed
  config_opencat['path_to_data_raw_oc20_s2ef_md'] = (
    config_opencat['path_to_data_raw_oc20'] + 's2ef_md/')
  config_opencat['path_to_data_raw_oc20_s2ef_rattled'] = (
    config_opencat['path_to_data_raw_oc20'] + 's2ef_rattled/')
    
  
  ### Set paths to IS2RE/S for oc22 ###
  config_opencat['path_to_data_raw_oc22_is2res'] = (
    config_opencat['path_to_data_raw_oc22'] + 
    'is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/')
  config_opencat['path_to_data_raw_oc22_is2res_train'] = (
    config_opencat['path_to_data_raw_oc22_is2res'] + 'train/')
  config_opencat['path_to_data_raw_oc22_is2res_val_id'] = (
    config_opencat['path_to_data_raw_oc22_is2res'] + 'val_id/')
  config_opencat['path_to_data_raw_oc22_is2res_val_ood'] = (
    config_opencat['path_to_data_raw_oc22_is2res'] + 'val_ood/')
  config_opencat['path_to_data_raw_oc22_is2res_test_id'] = (
    config_opencat['path_to_data_raw_oc22_is2res'] + 'test_id/')
  config_opencat['path_to_data_raw_oc22_is2res_test_ood'] = (
    config_opencat['path_to_data_raw_oc22_is2res'] + 'test_ood/')    
  
  
  ### Set paths to S2EF for oc22 ###
  config_opencat['path_to_data_raw_oc22_s2ef'] = (
    config_opencat['path_to_data_raw_oc22'] + 
    's2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/')
  config_opencat['path_to_data_raw_oc22_s2ef_train'] = (
    config_opencat['path_to_data_raw_oc22_s2ef'] + 'train/')
  config_opencat['path_to_data_raw_oc22_s2ef_val_id'] = (
    config_opencat['path_to_data_raw_oc22_s2ef'] + 'val_id/')
  config_opencat['path_to_data_raw_oc22_s2ef_val_ood'] = (
    config_opencat['path_to_data_raw_oc22_s2ef'] + 'val_ood/')
  config_opencat['path_to_data_raw_oc22_s2ef_test_id'] = (
    config_opencat['path_to_data_raw_oc22_s2ef'] + 'test_id/')
  config_opencat['path_to_data_raw_oc22_s2ef_test_ood'] = (
    config_opencat['path_to_data_raw_oc22_s2ef'] + 'test_ood/')   
  
  
  ### Set paths to miscellaneous meta data ###
  
  # set path to periodic table data
  config_opencat['path_to_data_raw_periodic_table'] = (
    config_opencat['path_to_data_raw'] + 'PubChemElements_all.csv')  
  
  # set path to Bader charge data for oc20
  config_opencat['path_to_data_raw_oc20_bader'] = (
    config_opencat['path_to_data_raw_oc20'] + 'oc20_bader_data/bader/')  
      
  # meta data for IS2RE/S test challenge in oc20
  config_opencat['path_to_data_raw_oc20_is2res_test_challenge_meta'] = (
    config_opencat['path_to_data_raw_oc20_is2res_test_challenge'] + 
    'metadata.npz')    
  
  # meta data for oc20
  config_opencat['path_to_data_raw_oc20_meta1'] = (
    config_opencat['path_to_data_raw_oc20'] + 'oc20_data_mapping.pkl')
  config_opencat['path_to_data_raw_oc20_meta2'] = (
    config_opencat['path_to_data_raw_oc20'] + 'mapping_adslab_slab.pkl')
  
  # meta data for oc22
  config_opencat['path_to_data_raw_oc22_meta1'] = (
    config_opencat['path_to_data_raw_oc22'] + 'oc22_metadata.pkl')
  config_opencat['path_to_data_raw_oc22_meta2'] = (
    config_opencat['path_to_data_raw_oc22'] + 'oc20_ref.pkl')
  
  
  ### Set paths to results ###
  
  config_opencat['path_to_data'] = (
    config['general']['path_to_data'] + 'OpenCatalyst/')
  config_opencat['path_to_data_metadata'] = (
    config_opencat['path_to_data'] + 'additional/')
  config_opencat['path_to_data_subtask'] = (
    config_opencat['path_to_data'] + '{}/'.format(subtask))
  config_opencat['path_to_data_subtask_train'] = (
    config_opencat['path_to_data_subtask'] + 'training/')
  config_opencat['path_to_data_subtask_val'] = (
    config_opencat['path_to_data_subtask'] + 'validation/')
  config_opencat['path_to_data_subtask_test'] = (
    config_opencat['path_to_data_subtask'] + 'testing/')
  
  
  # out of distribution test splitting rules in space
  # Stuctures have between 7 and 235 atoms. Calculate the bounds of atom numbers
  # that we want to be separated out for testing.
  min_atoms = 7
  max_atoms = 235
  middle_n_atoms = round((min_atoms + max_atoms) / 2)
  b_low = middle_n_atoms - round(
    config_opencat['spatial_test_split'] * (middle_n_atoms-min_atoms))
  b_high = middle_n_atoms + round(
    config_opencat['spatial_test_split'] *(max_atoms-middle_n_atoms)) 
  n_atoms_test_bounds = (b_low, b_high)
  
  # dictionary saving rules
  config_opencat['spatial_ood'] = {
    'n_atoms_test_bounds' : n_atoms_test_bounds
  }
  
  
  ### Create saving folder structure ###
  if save:
    
    # create directory if not existent for entire task    
    check_create_dir(config_opencat['path_to_data'])
    check_create_dir(config_opencat['path_to_data_metadata'])
    
    # delete any previous results for this subtask
    if os.path.isdir(config_opencat['path_to_data_subtask']):
      shutil.rmtree(config_opencat['path_to_data_subtask'])
    
    # iterate over all directories
    for path in [config_opencat['path_to_data_subtask'],
      config_opencat['path_to_data_subtask_train'],
      config_opencat['path_to_data_subtask_val'],
      config_opencat['path_to_data_subtask_test']]:
      
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
    

