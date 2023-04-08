import os
import pandas as pd

import selberai.data.download_data as download_data

def load (name: str, sample_only=False, path_to_data=None, token=None):
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
    
  # load data
  
