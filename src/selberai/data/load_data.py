import os
import pandas as pd
import numpy as np

import selberai.data.download_data as download_data

def load(name: str, sample_only=False, path_to_data=None, token=None) -> (
  np.array, np.array, np.array):
  """
  """
  
  # set standard path to data if not provided
  if path_to_data is None:
    path_to_data = 'datasets/{}/processed/'.format(name)
  
  # list directory of dataset
  dir_cont = os.listdir(path_to_data)
  
  # check if dataset available
  if ('training' in dir_cont and 'testing' in dir_cont and 
    'validation' in dir_cont):
    dir_cont_train = os.listdir(path_to_data+'training/')
    dir_cont_val = os.listdir(path_to_data+'validation/')
    dir_cont_test = os.listdir(path_to_data+'testing/')
    
    if (len(dir_cont_train) != 0 and len(dir_cont_val) != 0 and
      len(dir_cont_test) != 0):
      data_avail = True
    else:
      data_avail = False
      print('\nDataset directory exists, but some data is missing!\n')
      
  else:
    data_avail = False
    print('\nDataset is not available!\n')
    
  # download data if not available or missing files
  if not data_avail:
    download_data.download(name, path_to_data, token)
    
  # read the directories
  path_to_train = path_to_data + 'training/'
  path_to_val = path_to_data + 'validation/'
  path_to_test = path_to_data + 'testing/'
  train_cont = os.listdir(path_to_train)
  val_cont = os.listdir(path_to_val)
  test_cont = os.listdir(path_to_testing)
  
  # load data and conver to numpy
  train = pd.concat((pd.read_csv(path_to_train+f) for f in train_cont)).to_numpy()
  val = pd.concat((pd.read_csv(path_to_val+f) for f in val_cont)).to_numpy()
  test = pd.concat((pd.read_csv(path_to_test+f) for f in test_cont)).to_numpy()
  
  # set and return value
  return_value = (train, val, test)
  return return_value
  
  
