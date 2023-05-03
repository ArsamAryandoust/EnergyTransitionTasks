import os
import gc
import json
import random

from tqdm import tqdm
import ocpmodels
from config.load_oc import config_OC
from ocpmodels.datasets import TrajectoryLmdbDataset, SinglePointLmdbDataset


def process_all_datasets(config: dict, save: bool):
  """
  """
  print("Processing Open Catalyst dataset.")
  
  # iterate over all subtasks
  for subtask in config['OpenCatalyst']['subtask_list']:
  
    # augment conigurations with additional information
    config_opencat = config_OC(config, subtask, save)
    
    # do data processing according to current subtask
    if subtask == 'oc20_s2ef':
      config_opencat['subsample_frac'] *= 0.1
      process_oc20_s2ef(config_opencat, save)
      
    elif subtask == 'oc20_is2res':
      process_oc20_is2res(config_opencat, save)
      
    elif subtask == 'oc22_s2ef':
      process_oc22_s2ef(config_opencat, save)
      
    elif subtask == 'oc22_is2res':
      process_oc22_is2res(config_opencat, save)
      
      
      
def load_dataset(path:str, single_lmdb:bool=False) -> (
  ocpmodels.datasets.lmdb_dataset.SinglePointLmdbDataset | 
  ocpmodels.datasets.lmdb_dataset.TrajectoryLmdbDataset):
  """
  """
  
  if single_lmdb:
    filename = os.listdir(path)[1]
    path += filename
    dataset = SinglePointLmdbDataset({"src": path})
    
  else:
    dataset = TrajectoryLmdbDataset({"src": path})
      
  return dataset    
  
  
  
def create_is2res_data(config_opencat: dict, path_list: list[str], 
  single_lmdb=False):
  """
  """

  # create empty dictionary to save results
  is2res_data_dict = {}
  id_counter = 0
  
  # iterate over passed path list
  for path in path_list:
  
    # load dataset from currently iterated path
    is2res_dataset = load_dataset(path, single_lmdb)
    
    # subsample data
    n_data = len(is2res_dataset)
    n_samples = int(config_opencat['subsample_frac'] * n_data)
    random.seed(config_opencat['seed'])
    sample_list = random.sample(range(n_data), n_samples)
    
    # iterate over all datapoints
    for data_index in tqdm(sample_list):
      
      # get random datapoint
      datapoint = is2res_dataset[data_index]
      
      # save datapoint information in results dictionary
      is2res_data_dict[id_counter] = {
        'relexed_energy' : datapoint.y_relaxed,
        'atoms' : datapoint.atomic_numbers.int().tolist(),
        'initial_structure' : datapoint.pos.tolist(),
        'relaxed_strucutre' : datapoint.pos_relaxed.tolist()
      }
      
      # increment ID counter
      id_counter += 1
      
  # free up memory
  del is2res_dataset
  gc.collect()
  
  return is2res_data_dict
  
    
    
def create_s2ef_data(config_opencat: dict, path_list: list[str]):
  """
  """
  
  # create empty dictionary to save results
  s2ef_data_dict = {}
  id_counter = 0
  
  # iterate over passed path list
  for path in path_list:
  
    # load dataset from currently iterated path
    s2ef_dataset = load_dataset(path)
    
    # subsample data
    n_data = len(s2ef_dataset)
    n_samples = int(config_opencat['subsample_frac'] * n_data)
    random.seed(config_opencat['seed'])
    sample_list = random.sample(range(n_data), n_samples)
    
    # iterate over all datapoints
    for data_index in tqdm(sample_list):
      
      # get random datapoint
      datapoint = s2ef_dataset[data_index]
      
      # save datapoint information in results dictionary
      s2ef_data_dict[id_counter] = {
        'energy' : datapoint.y,
        'atoms' : datapoint.atomic_numbers.int().tolist(),
        'structure' : datapoint.pos.tolist(),
        'forces' : datapoint.force.tolist()
      }
      
      # increment ID counter
      id_counter += 1
      
  # free up memory
  del s2ef_dataset
  gc.collect()
  
  return s2ef_data_dict
  
  
  
def save_data(config_opencat: dict, data_dict: dict, path_to_folder: str, 
  name_base: str):
  """
  """
  
  # create shuffled list of ordered integers
  random_list = list(range(len(data_dict)))
  random.seed(config_opencat['seed'])
  random.shuffle(random_list)
  
  # create list of data_per_file sized samples
  sample_list = []
  for i in range(0, len(random_list), config_opencat['data_per_file']):
    sample_list.append(random_list[i:i + config_opencat['data_per_file']])
  

  # iterate over list of lists
  for index, samples in enumerate(sample_list):
    
    # get subset of dictionary entries
    save_dict = dict((k, data_dict[k]) for k in samples)
    
    # set path for saving
    filename = name_base + '_' + str(index+1) + '.json'
    saving_path = path_to_folder + filename
    
    # save to json file
    with open(saving_path, 'w') as w_file:
      json.dump(save_dict, w_file)
  
  
  
def process_oc20_s2ef(config_opencat: dict, save: bool):
  """
  """  
  
  # create all relevant data paths
  p_train = config_opencat['path_to_data_raw_oc20_s2ef_train']
  p_val_id = config_opencat['path_to_data_raw_oc20_s2ef_val_id']
  p_val_ood_ads = config_opencat['path_to_data_raw_oc20_s2ef_val_ood_ads']
  p_val_ood_cat = config_opencat['path_to_data_raw_oc20_s2ef_val_ood_cat']
  p_val_ood_both = config_opencat['path_to_data_raw_oc20_s2ef_val_ood_both']

  
  # Create training data from all in distribution datasets
  s2ef_data_dict = create_s2ef_data(config_opencat, [p_train])
  
  if save:
    save_data(config_opencat, s2ef_data_dict, 
      config_opencat['path_to_data_subtask_train'], 'training')
  
  # Create validation data from all in distribution datasets
  s2ef_data_dict = create_s2ef_data(config_opencat, [p_val_id])
  
  if save:
    save_data(config_opencat, s2ef_data_dict, 
      config_opencat['path_to_data_subtask_val'], 'validation')
  
  # create testing data from out-of-distribution datasets
  s2ef_data_dict = create_s2ef_data(config_opencat, 
    [p_val_ood_ads, p_val_ood_cat, p_val_ood_both])
  
  if save:
    save_data(config_opencat, s2ef_data_dict, 
      config_opencat['path_to_data_subtask_test'], 'testing')
  
  
  
def process_oc20_is2res(config_opencat: dict, save: bool):
  """
  """
  
  # create all relevant data paths
  p_train = config_opencat['path_to_data_raw_oc20_is2res_train']
  p_val_id = config_opencat['path_to_data_raw_oc20_is2res_val_id']
  p_val_ood_ads = config_opencat['path_to_data_raw_oc20_is2res_val_ood_ads']
  p_val_ood_cat = config_opencat['path_to_data_raw_oc20_is2res_val_ood_cat']
  p_val_ood_both = config_opencat['path_to_data_raw_oc20_is2res_val_ood_both']


  # Create training data from all in distribution datasets
  is2res_data_dict = create_is2res_data(config_opencat, 
    [p_train], single_lmdb=True)
    
  if save:
    save_data(config_opencat, is2res_data_dict, 
      config_opencat['path_to_data_subtask_train'], 'training')
    
  # Create validation data from all in distribution datasets
  is2res_data_dict = create_is2res_data(config_opencat, 
    [p_val_id], single_lmdb=True)
  
  if save:
    save_data(config_opencat, is2res_data_dict, 
      config_opencat['path_to_data_subtask_val'], 'validation')
  
  # create testing data from out-of-distribution datasets
  is2res_data_dict = create_is2res_data(config_opencat, 
    [p_val_ood_ads, p_val_ood_cat, p_val_ood_both], single_lmdb=True)
    
  if save:
    save_data(config_opencat, is2res_data_dict, 
      config_opencat['path_to_data_subtask_test'], 'testing')
  
  
  
def process_oc22_s2ef(config_opencat: dict, save: bool):
  """
  """
  
  # create all relevant data paths
  p_train = config_opencat['path_to_data_raw_oc22_s2ef_train']
  p_val_id = config_opencat['path_to_data_raw_oc22_s2ef_val_id']
  p_val_ood = config_opencat['path_to_data_raw_oc22_s2ef_val_ood']

  # Create training data from all in distribution datasets
  s2ef_data_dict = create_s2ef_data(config_opencat, [p_train])
  
  if save:
    save_data(config_opencat, s2ef_data_dict, 
      config_opencat['path_to_data_subtask_train'], 'training')
  
  # Create validation data from all in distribution datasets
  s2ef_data_dict = create_s2ef_data(config_opencat, [p_val_id])
  
  if save:
    save_data(config_opencat, s2ef_data_dict, 
      config_opencat['path_to_data_subtask_val'], 'validation')
      
  # create validation data from out-of-distribution datasets
  s2ef_data_dict = create_s2ef_data(config_opencat, [p_val_ood])
  
  if save:
    save_data(config_opencat, s2ef_data_dict, 
      config_opencat['path_to_data_subtask_test'], 'testing')
      
      
  
def process_oc22_is2res(config_opencat: dict, save: bool):
  """
  """
  
  # create all relevant data paths
  p_train = config_opencat['path_to_data_raw_oc22_is2res_train']
  p_val_id = config_opencat['path_to_data_raw_oc22_is2res_val_id']
  p_val_ood = config_opencat['path_to_data_raw_oc22_is2res_val_ood']
  
  # Create training data from all in distribution datasets
  is2res_data_dict = create_is2res_data(config_opencat, [p_train])
  
  if save:
    save_data(config_opencat, is2res_data_dict, 
      config_opencat['path_to_data_subtask_train'], 'training')
  
  # Create training data from all in distribution datasets
  is2res_data_dict = create_is2res_data(config_opencat, [p_val_id])
  
  if save:
    save_data(config_opencat, is2res_data_dict, 
      config_opencat['path_to_data_subtask_val'], 'validation')
  
  # create validation data from out-of-distribution datasets
  is2res_data_dict = create_is2res_data(config_opencat, [p_val_ood])
  
  if save:
    save_data(config_opencat, is2res_data_dict, 
      config_opencat['path_to_data_subtask_test'], 'testing')
  
  
  
  
