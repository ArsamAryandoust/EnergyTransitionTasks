import os
from tqdm import tqdm
import pandas as pd
import numpy as np

import selberai.data.download_data as download_data

class Dataset:
  """
  """
  
  def __init__(self, train: np.array, val: np.array, test: np.array, 
    add: pd.DataFrame):
    
    self.train = train
    self.val = val
    self.test = test
    self.add = add
    

def load(name: str, subtask: str, sample_only=False, tabular=False, 
  path_to_data=None, path_to_token=None) -> Dataset:
  """
  """
  
  # set standard path to data if not provided
  if path_to_data is None:
    path_to_data = 'datasets/{}/processed/'.format(name)
  else:
    path_to_data += name + '/'
  
  # extend path 
  path_to_data += subtask + '/'
  
  # list directory of dataset
  dir_cont = set(os.listdir(path_to_data))
  
  # set paths and read the directories
  path_to_train = path_to_data + 'training/'
  path_to_val = path_to_data + 'validation/'
  path_to_test = path_to_data + 'testing/'
  
  # check if dataset available
  if ('training' in dir_cont and 'testing' in dir_cont and 
    'validation' in dir_cont):
    
    # read content of available train, val and test directories
    dir_cont_train = os.listdir(path_to_train)
    dir_cont_val = os.listdir(path_to_val)
    dir_cont_test = os.listdir(path_to_test)
    
    # check if content is non-zero
    if (len(dir_cont_train) != 0 and len(dir_cont_val) != 0 and
      len(dir_cont_test) != 0):
      data_avail = True
    
    else:
      data_avail = False
      print('\nDataset available, but some datasets are missing files!\n')
    
  else:
    data_avail = False
    print('\nDataset is not available on {}!\n'.format(path_to_data))
  
  # download data if not available or missing files
  if not data_avail:
    # TO DO: implement download of subtask data only
    path_to_data = path_to_data[:-(len(subtask)+1)]
    download_data.download(name, subtask, path_to_data, path_to_token)
  
  
  ###
  # Load training, validation and testing ###
  ###
  
  # read directory content
  train_cont = os.listdir(path_to_train)
  val_cont = os.listdir(path_to_val)
  test_cont = os.listdir(path_to_test)
  
  # reduce directory content to single file only if sample_only==True
  if sample_only:
    train_cont = train_cont[:1]
    val_cont = val_cont[:1]
    test_cont = test_cont[:1]
  
  # set empty dataframes
  train = pd.DataFrame()
  val = pd.DataFrame()
  test = pd.DataFrame()
  
  # iterate over train and concatenate
  print("Loading training data.")
  pbar = tqdm(total=len(train_cont))
  for f_name in train_cont:
    train = pd.concat((train, pd.read_csv(path_to_train+f_name)))
    pbar.update(1)
    
  # iterate over val and concatenate
  print("Loading validation data.")
  pbar = tqdm(total=len(val_cont))
  for f_name in val_cont:
    val = pd.concat((val, pd.read_csv(path_to_val+f_name)))
    pbar.update(1)
    
  # iterate over test and concatenate
  print("Loading testing data.")
  pbar = tqdm(total=len(test_cont))
  for f_name in test_cont:
    test = pd.concat((test, pd.read_csv(path_to_test+f_name)))
    pbar.update(1)
  
  
  ###
  # Convert to unified data representation and potentially load additional ###
  ###
  
  # convert BuildingElectricity to unified representation
  if name == 'BuildingElectricity':
    path = path_to_data+'additional/id_histo_map.csv'
    add = {'id_histo_map': pd.read_csv(path)}
    if not tabular:
      train, val, test = convert_be(train), convert_be(val), convert_be(test)
  
  # convert WindFarm to unified representation
  elif name == 'WindFarm':
    add = None
    if not tabular:
      train, val, test = convert_wf(train), convert_wf(val), convert_wf(test)
      
  # convert ClimART to unified representation
  elif name == 'ClimART':
    add = None
    if not tabular:
      train = convert_ca(train, subtask)
      val = convert_ca(val, subtask)
      test = convert_ca(test, subtask)
      
  # set and return values as Dataset object
  dataset = Dataset(train, val, test, add)
  
  return dataset
  
  
def convert_ca(dataframe: pd.DataFrame, subtask: str) -> dict:
  """
  """
  # set values from config file
  n_data = len(dataframe.index)
  n_layers = 49
  n_levels = 50
  vars_global = 79 # 82 globals. 3 coordinates x,y,z extracted. are in x_s
  vars_levels = 4
  if subtask == 'pristine':
    vars_layers = 14
  elif subtask == 'clear_sky':
    vars_layers = 45
  
  # set starting and end indices of tabular features
  end_t = 2
  end_s = end_t + 3
  end_st_1 = end_s + vars_global
  end_st_2 = end_st_1 + vars_layers * n_layers
  end_st_3 = end_st_2 + vars_levels * n_levels
  end_y1 = end_st_3 + 2 * n_layers
  
  data_dict = {}
  data_dict['x_t'] = dataframe.iloc[:, :end_t].to_numpy()
  data_dict['x_s'] = dataframe.iloc[:, end_t:end_s].to_numpy()
  data_dict['x_st_1'] = dataframe.iloc[:, end_s:end_st_1].to_numpy()
  data_dict['x_st_2'] = dataframe.iloc[:, end_st_1:end_st_2].to_numpy()
  data_dict['x_st_3'] = dataframe.iloc[:, end_st_2:end_st_3].to_numpy()
  data_dict['y_st_1'] = dataframe.iloc[:, end_st_3:end_y1].to_numpy()
  data_dict['y_st_2'] = dataframe.iloc[:, end_y1:].to_numpy()
  
  # reshape arrays
  data_dict['x_st_2'] = np.reshape(data_dict['x_st_2'], 
    (n_data, n_layers, vars_layers), order='C')
  data_dict['x_st_3'] = np.reshape(data_dict['x_st_3'], 
    (n_data, n_levels, vars_levels), order='C')
  data_dict['y_st_1'] = np.reshape(data_dict['y_st_1'], 
    (n_data, n_layers, 2), order='C')
  data_dict['y_st_2'] = np.reshape(data_dict['y_st_2'], 
    (n_data, n_levels, 4), order='C')
  
  return data_dict
  
  
def convert_wf(dataframe: pd.DataFrame) -> dict:
  """
  """
  # set values from config file
  hist_window = 288
  pred_window = 288
  n_times = 3
  n_states = 10
  n_data = len(dataframe.index)
  
  # set starting and end indices of tabular features
  end_s = 2
  end_t1 = end_s + hist_window * n_times
  end_st = end_t1 + hist_window * n_states
  end_t2 = end_st + pred_window * n_times
  
  # fill dataset dictionary
  data_dict = {}
  data_dict['x_s'] = dataframe.iloc[:, :end_s].to_numpy()
  data_dict['x_t_1'] = dataframe.iloc[:, end_s:end_t1].to_numpy().astype(int)
  data_dict['x_st'] = dataframe.iloc[:, end_t1:end_st].to_numpy()
  data_dict['x_t_2'] = dataframe.iloc[:, end_st:end_t2].to_numpy().astype(int)
  data_dict['y_st'] = dataframe.iloc[:, end_t2:].to_numpy()
  
  # either order='C' with shape (n_data, n_states, hist_window)
  # or order='F' with shape (n_data, hist_window, n_states)
  data_dict['x_t_1'] = np.reshape(data_dict['x_t_1'], 
    (n_data, n_times, hist_window), order='C')
  data_dict['x_t_2'] = np.reshape(data_dict['x_t_2'], 
    (n_data, n_times, pred_window), order='C')
  data_dict['x_st'] = np.reshape(data_dict['x_st'], 
    (n_data, n_states, hist_window), order='C')
  
  return data_dict
  
  
def convert_be(dataframe: pd.DataFrame) -> dict:
  """
  """
  # set values from config file
  hist_window = 24
  n_states = 9
  n_data = len(dataframe.index)
  
  # set starting and end indices of tabular features
  end_t = 5
  start_st = end_t + 1
  end_st = start_st + hist_window * n_states
  
  # fill dataset dictionary
  data_dict = {}
  data_dict['x_t'] = dataframe.iloc[:, :end_t].to_numpy().astype(int)
  data_dict['x_s'] = dataframe.iloc[:, end_t].to_numpy().astype(int)
  data_dict['x_st'] = dataframe.iloc[:, start_st:end_st].to_numpy()
  data_dict['y_st'] = dataframe.iloc[:, end_st:].to_numpy()
  
  # either order='C' with shape (n_data, n_states, hist_window)
  # or order='F' with shape (n_data, hist_window, n_states)
  data_dict['x_st'] = np.reshape(data_dict['x_st'], 
    (n_data, n_states, hist_window), order='C')
  
  return data_dict
  
  
