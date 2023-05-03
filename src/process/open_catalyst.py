import os
import gc
import json

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
      process_oc20_s2ef(config_opencat, save)
      
    elif subtask == 'oc20_is2res':
      process_oc20_is2res(config_opencat, save)
      
    elif subtask == 'oc22_s2ef':
      process_oc22_s2ef(config_opencat, save)
      
    elif subtask == 'oc22_is2res':
      process_oc22_is2res(config_opencat, save)



def load_dataset(path:str, single_point_lmdb:bool=False) -> (
  ocpmodels.datasets.lmdb_dataset.SinglePointLmdbDataset | 
  ocpmodels.datasets.lmdb_dataset.TrajectoryLmdbDataset):
  """
  """
  
  if single_point_lmdb:
    filename = os.listdir(path)[1]
    path += filename
    dataset = SinglePointLmdbDataset({"src": path})
    
  else:
    dataset = TrajectoryLmdbDataset({"src": path})
      
  return dataset    
  
  
  
def create_is2res_data(config_opencat: dict, dataset_list: (
  list[ocpmodels.datasets.lmdb_dataset.SinglePointLmdbDataset] |
  list[ocpmodels.datasets.lmdb_dataset.TrajectoryLmdbDataset]), save: bool):
  """
  """


def create_s2ef_data(config_opencat: dict, path_list: list[str], save: bool):
  """
  """
  
  # create empty dictionary to save results
  s2ef_data_dict = {}
  id_counter = 0
  
  # iterate over passed path list
  for path in path_list:
  
    # load dataset from currently iterated path
    s2ef_dataset = load_dataset(path)
    
    # iterate over all datapoints
    for datapoint in tqdm(s2ef_dataset):
      
      # save datapoint information in results dictionary
      s2ef_data_dict[id_counter] = {
        'structure' : datapoint.pos,
        'energy' : datapoint.y,
        'forces' : datapoint.force
      }
  
      
  # free up memory
  del s2ef_dataset
  gc.collect()
  
  # save to json 
  if save:
    pass
  
def process_oc20_s2ef(config_opencat: dict, save: bool):
  """
  """  
  
  # create all relevant data paths
  train_path = config_opencat['path_to_data_raw_oc20_s2ef_train']
  val_id_path = config_opencat['path_to_data_raw_oc20_s2ef_val_id']
  val_ood_ads_path = config_opencat['path_to_data_raw_oc20_s2ef_val_ood_ads']
  val_ood_cat_path = config_opencat['path_to_data_raw_oc20_s2ef_val_ood_cat']
  val_ood_both_path = config_opencat['path_to_data_raw_oc20_s2ef_val_ood_both']
  test_id_path = config_opencat['path_to_data_raw_oc20_s2ef_test_id']
  test_ood_ads_path = config_opencat['path_to_data_raw_oc20_s2ef_test_ood_ads']
  test_ood_cat_path = config_opencat['path_to_data_raw_oc20_s2ef_test_ood_cat']
  test_ood_both_path = config_opencat['path_to_data_raw_oc20_s2ef_test_ood_both']

  
  # Create training data from all in distribution datasets
  create_s2ef_data(config_opencat, [train_path, val_id_path, test_id_path], 
    save)
  
  # create validation data from out-of-distribution datasets
  create_s2ef_data(config_opencat, 
    [val_ood_ads_path, val_ood_cat_path, val_ood_both_path], save)
  
  # create testing data from out-of-distribution datasets
  create_s2ef_data(config_opencat, 
    [test_ood_ads_path, test_ood_cat_path, test_ood_both_path], save)
  
  
  
def process_oc20_is2res(config_opencat: dict, save: bool):
  """
  """
  
  # create all relevant data paths
  train_path = config_opencat['path_to_data_raw_oc20_is2res_train']
  val_id_path = config_opencat['path_to_data_raw_oc20_is2res_val_id']
  val_ood_ads_path = config_opencat['path_to_data_raw_oc20_is2res_val_ood_ads']
  val_ood_cat_path = config_opencat['path_to_data_raw_oc20_is2res_val_ood_cat']
  val_ood_both_path = config_opencat['path_to_data_raw_oc20_is2res_val_ood_both']
  test_id_path = config_opencat['path_to_data_raw_oc20_is2res_test_id']
  test_ood_ads_path = config_opencat['path_to_data_raw_oc20_is2res_test_ood_ads']
  test_ood_cat_path = config_opencat['path_to_data_raw_oc20_is2res_test_ood_cat']
  test_ood_both_path = config_opencat['path_to_data_raw_oc20_is2res_test_ood_both']
  test_challenge_path = config_opencat['path_to_data_raw_oc20_is2res_test_challenge']


  # load all datasets
  train_dataset = load_dataset(train_path, single_point_lmdb=True)
  val_id_dataset = load_dataset(val_id_path, single_point_lmdb=True)
  val_ood_ads_dataset = load_dataset(val_ood_ads_path, single_point_lmdb=True)
  val_ood_cat_dataset = load_dataset(val_ood_cat_path, single_point_lmdb=True)
  val_ood_both_dataset = load_dataset(val_ood_both_path, single_point_lmdb=True)
  test_id_dataset = load_dataset(test_id_path, single_point_lmdb=True)
  test_ood_ads_dataset = load_dataset(test_ood_ads_path, single_point_lmdb=True)
  test_ood_cat_dataset = load_dataset(test_ood_cat_path, single_point_lmdb=True)
  test_ood_both_dataset = load_dataset(test_ood_both_path, single_point_lmdb=True)
  test_challenge_dataset = load_dataset(test_challenge_path, single_point_lmdb=True)
  
  
  
def process_oc22_s2ef(config_opencat: dict, save: bool):
  """
  """
  
  train_path = config_opencat['path_to_data_raw_oc22_s2ef_train']
  val_id_path = config_opencat['path_to_data_raw_oc22_s2ef_val_id']
  val_ood_path = config_opencat['path_to_data_raw_oc22_s2ef_val_ood']
  test_id_path = config_opencat['path_to_data_raw_oc22_s2ef_test_id']
  test_ood_path = config_opencat['path_to_data_raw_oc22_s2ef_test_ood']


  # load all datasets
  train_dataset = load_dataset(train_path)
  val_id_dataset = load_dataset(val_id_path)
  val_ood_dataset = load_dataset(val_ood_path)
  test_id_dataset = load_dataset(test_id_path)
  test_ood_dataset = load_dataset(test_ood_path)
  
  
  
  
def process_oc22_is2res(config_opencat: dict, save: bool):
  """
  """
  
  # create all relevant data paths
  train_path = config_opencat['path_to_data_raw_oc22_is2res_train']
  val_id_path = config_opencat['path_to_data_raw_oc22_is2res_val_id']
  val_ood_path = config_opencat['path_to_data_raw_oc22_is2res_val_ood']
  test_id_path = config_opencat['path_to_data_raw_oc22_is2res_test_id']
  test_ood_path = config_opencat['path_to_data_raw_oc22_is2res_test_ood']
  
  # load all datasets
  train_dataset = load_dataset(train_path, single_point_lmdb=True)
  val_id_dataset = load_dataset(val_id_path, single_point_lmdb=True)
  val_ood_dataset = load_dataset(val_ood_path, single_point_lmdb=True)
  test_id_dataset = load_dataset(test_id_path, single_point_lmdb=True)
  test_ood_dataset = load_dataset(test_ood_path, single_point_lmdb=True)
  
  
  
  
  
  
