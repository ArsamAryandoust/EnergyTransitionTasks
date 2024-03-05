import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor

import selberai.data.download_data as download_data

class Dataset:
  """
  """
  
  def __init__(self, 
    train: (dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | 
      tuple[pd.DataFrame, pd.DataFrame]), 
    val: (dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | 
      tuple[pd.DataFrame, pd.DataFrame]),
    test: (dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | 
      tuple[pd.DataFrame, pd.DataFrame]), 
    add: dict[str, np.ndarray] | None):
    
    
    self.train = train
    self.val = val
    self.test = test
    self.add = add
    

def load(name: str, subtask: str, sample_only: bool=False, form: str='uniform', 
  path_to_data: str=None, path_to_token: str=None) -> Dataset:
  """
  Loads the data into memory and returns it.
  
  Parameters:
   - name (str): full name of the dataset to load.
   - subtask (str): subtask of the dataset to load.
   - sample_only (bool): if True, loads only 1 csv file for each split. 
   Default: False.
   - form (str): determines the output format. Can be 'uniform', 'tabular' or 
   'dataframe'. Default: 'uniform'
   - path_to_data (str):  path to the data folder containing all datasets.
   - path_to_token (str): path to the token for https://dataverse.harvard.edu

  Returns a Dataset object with the attributes 'train', 'val', 'test' and 'add'.
  
  If the 'form' parameter is 'uniform', each attribute contains a dictionary 
  with x_t, x_s, x_st, y data depending on the dataset.

  If the 'form' parameter is 'tabular', each split contains a tuple with the 
  full features and labels in np.ndarray format, while 'add' contains dict.
  
  if the 'form' parameter is 'dataframe', each attribute contains a single 
  pandas dataframe with all features and labels.
  """
  
  ###
  # Set some paths and make sure data is available ###
  ###
  
  if path_to_data is None:
    path_to_data = 'datasets/{}/processed/'.format(name)
    
  else:
    path_to_data += name + '/'
  
  # check if data is available
  check_and_download_data(name, subtask, path_to_data, path_to_token)
  
  # extend path for subtask
  if (subtask == 'oc20_is2rs' or subtask == 'oc22_is2rs' or 
    subtask == 'oc20_is2re' or subtask == 'oc22_is2re'):
    path_to_data_subtask = path_to_data + subtask[:-1] + 'es/'
  else:
    path_to_data_subtask = path_to_data + subtask + '/'
  
  ###
  # Load training, validation and testing ###
  ###

  # set paths and read the directories
  path_to_train = path_to_data_subtask + 'training/'
  path_to_val = path_to_data_subtask + 'validation/'
  path_to_test = path_to_data_subtask + 'testing/'
    
  # read directory content
  train_cont = os.listdir(path_to_train)
  val_cont = os.listdir(path_to_val)
  test_cont = os.listdir(path_to_test)
  
  # reduce directory content to single file only if sample_only==True
  if sample_only:
    train_cont = train_cont[:1]
    val_cont = val_cont[:1]
    test_cont = test_cont[:1]
  
  # Import data
  print("\nLoading dataset splits for train, validation and testing:")
  load_func = None
  if name == 'OpenCatalyst':
    load_func = load_json_fast
    if form != "uniform":
      raise ValueError("Tabular and DataFrame return formats are not implemented yet, \nPlease choose form=uniform to load Open Catalyst data.")
  else:
    load_func = load_csv_fast

  train = load_func(path_to_train, train_cont)
  val = load_func(path_to_val, val_cont)
  test = load_func(path_to_test, test_cont)
  
    
  
  ###
  # Convert to unified data representation and potentially load additional ###
  ###
  
  if name == 'BuildingElectricity':
  
    train = convert_be(train, form)
    val = convert_be(val, form)
    test = convert_be(test, form)
    
    add = load_add_be(path_to_data_subtask, form)


  elif name == 'UberMovement':
  
    train = convert_um(train, form)
    val = convert_um(val, form)
    test = convert_um(test, form)
    
    add = None
  
  
  elif name == 'WindFarm':
  
    train = convert_wf(train, form)
    val = convert_wf(val, form)
    test = convert_wf(test, form)
    
    add = None
      
      
  elif name == 'ClimART':
  
    train = convert_ca(train, subtask, form)
    val = convert_ca(val, subtask, form)
    test = convert_ca(test, subtask, form)
    
    add = None
      
      
  elif name == 'OpenCatalyst':
    
    train = convert_oc(train, subtask, form)
    val = convert_oc(val, subtask, form)
    test = convert_oc(test, subtask, form)
    
    add = load_add_oc(path_to_data, form)
    
    
      
  elif name == 'Polianna':
    
    train = convert_pa(train, subtask, form)
    val = convert_pa(val, subtask, form)
    test = convert_pa(test, subtask, form)
    
    add = load_add_pa(path_to_data_subtask, subtask)
        
  ### Set and return values as Dataset object ###
  dataset = Dataset(train, val, test, add)
  
  return dataset
  

def check_and_download_data(name: str, subtask: str, path_to_data: str, 
  path_to_token: str):
  """
  """
  
  # extend path for subtask
  if (subtask == 'oc20_is2rs' or subtask == 'oc22_is2rs' or 
    subtask == 'oc20_is2re' or subtask == 'oc22_is2re'):
    path_to_data_subtask = path_to_data + subtask[:-1] + 'es/'
  else:
    path_to_data_subtask = path_to_data + subtask + '/'
  
  # list directory of dataset
  dir_cont = set(os.listdir(path_to_data_subtask))
  
  # check if dataset available
  if ('training' in dir_cont and 'testing' in dir_cont and 
    'validation' in dir_cont):
    
    # set paths and read the directories
    path_to_train = path_to_data_subtask + 'training/'
    path_to_val = path_to_data_subtask + 'validation/'
    path_to_test = path_to_data_subtask + 'testing/'
    
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
    print('\nDataset is not available on {}!\n'.format(path_to_data_subtask))
    
    
  # download data if not available or missing files
  if not data_avail:
    download_data.download(name, subtask, path_to_data, path_to_token)
    # TO DO: implement download of subtask data only
    

def convert_pa(df: pd.DataFrame, subtask: str, form: str) -> (
  dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | 
  tuple[pd.DataFrame, pd.DataFrame]):
  """
  """
  
  # set starting and end indices of tabular features
  end_t = 3
  end_s = end_t + 2
  end_st = end_s + 1
  

  if form == 'uniform':
    data_dict = {}
    data_dict['x_t'] = df.iloc[:, :end_t].to_numpy()
    data_dict['x_s'] = df.iloc[:, end_t:end_s].to_numpy()
    data_dict['x_st'] = df.iloc[:, end_s:end_st].to_numpy()
    if subtask == 'article_level':
      data_dict['y_st'] = df.iloc[:, end_st:].to_numpy()
    elif subtask == 'text_level':
      data_dict['y_st'] = data_dict['x_st']
    
    return data_dict
  
  elif form == 'tabular':
    features = df.iloc[:, :end_st].to_numpy()
    if subtask == 'article_level':
      labels = df.iloc[:, end_st:].to_numpy()
    elif subtask == 'text_level':
      labels = df.iloc[:, end_s:end_st].to_numpy()

    return features, labels
    
  elif form == 'dataframe':
    features = df.iloc[:, :end_st]
    if subtask == 'article_level':
      labels = df.iloc[:, end_st:]
    elif subtask == 'text_level':
      labels = df.iloc[:, end_s:end_st]

    return features, labels
  
  
def load_add_pa(path_to_data_subtask: str, subtask: str):
  """
  """
    
  path = path_to_data_subtask + 'additional/article_tokenized.json'
  add = {}
  
  # load article data
  with open(path, 'r') as json_file:
    add['x_st'] = json.load(json_file)
  
  # load label data
  if subtask == 'text_level':
    path = path_to_data_subtask + 'additional/annotation_labels.json'
    with open(path, 'r') as json_file:
      add['y_st'] = json.load(json_file)
  
  return add
  
  
def convert_oc(dict_dataset: dict, subtask: str, form: str) -> dict:
  """
  """
  
  if form == 'uniform':
    
    if subtask == 'oc20_is2re' or subtask == 'oc22_is2re':
      dict_x_s = {}
      dict_x_st = {}
      dict_y_t = {}
      for index, item in enumerate(dict_dataset.items()):
        dict_x_s[index] = item[1]['atomic_numbers']
        dict_x_st[index] = np.array(item[1]['initial_structure'])
        dict_y_t[index] = item[1]['relexed_energy']
      
      data_dict = {
        'x_s': dict_x_s,
        'x_st': dict_x_st,
        'y_t': dict_y_t
      }
      
    elif subtask == 'oc20_is2rs' or subtask == 'oc22_is2rs':
      dict_x_s = {}
      dict_x_st = {}
      dict_y_st = {}
      for index, item in enumerate(dict_dataset.items()):
        dict_x_s[index] = item[1]['atomic_numbers']
        dict_x_st[index] = np.array(item[1]['initial_structure'])
        dict_y_st[index] = np.array(item[1]['relaxed_strucutre'])
        
      data_dict = {
        'x_s': dict_x_s,
        'x_st': dict_x_st,
        'y_st': dict_y_st
      }
      
    elif subtask == 'oc20_s2ef' or subtask == 'oc22_s2ef':
      dict_x_s = {}
      dict_x_st = {}
      dict_y_t = {}
      dict_y_st = {}
      for index, item in enumerate(dict_dataset.items()):
        dict_x_s[index] = item[1]['atomic_numbers']
        dict_x_st[index] = np.array(item[1]['structure'])
        dict_y_t[index] = item[1]['energy']
        dict_y_st[index] = np.array(item[1]['forces'])
        
      data_dict = {
        'x_s': dict_x_s,
        'x_st': dict_x_st,
        'y_t': dict_y_t,
        'y_st': dict_y_st
      }
      
    return data_dict
  
  else:
    raise ValueError("Tabular and DataFrame return formats are not implemented yet, \nPlease choose form=uniform to load Open Catalyst data.")

  
  
def load_add_oc(path_to_data: str, form: str):
  """
  """
  
  # set paths
  path_x_s = path_to_data + 'additional/periodic_table.csv'
  path_x_s_1 = path_to_data + 'additional/numeric_feat.csv'
  path_x_s_2 = path_to_data + 'additional/ordinal_feat.csv'
  path_x_s_3 = path_to_data + 'additional/onehot_ox_feat.csv'
  
  # load data
  x_s = pd.read_csv(path_x_s)
  x_s_1 = pd.read_csv(path_x_s_1)
  x_s_2 = pd.read_csv(path_x_s_2)
  x_s_3 = pd.read_csv(path_x_s_3)
  
  # transform to numpy arrays if from != 'dataframe'  
  if form == 'uniform' or form == 'tabular':
    x_s = x_s.iloc[:, 1:].to_numpy()
    x_s_1 = x_s_1.iloc[:, 1:].to_numpy()
    x_s_2 = x_s_2.iloc[:, 1:].to_numpy()
    x_s_3 = x_s_3.iloc[:, 1:].to_numpy()
  
  # set return value
  add = {
    'x_s': x_s,
    'x_s_1': x_s_1,
    'x_s_2': x_s_2,
    'x_s_3': x_s_3
  }
  
  return add
  
  
def convert_ca(df: pd.DataFrame, subtask: str, form: str) -> (
  dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | 
  tuple[pd.DataFrame, pd.DataFrame]):
  """
  """
  
  # set values from config file
  n_data = len(df.index)
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
  
  if form == 'uniform':
    data_dict = {}
    data_dict['x_t'] = df.iloc[:, :end_t].to_numpy()
    data_dict['x_s'] = df.iloc[:, end_t:end_s].to_numpy()
    data_dict['x_st_1'] = df.iloc[:, end_s:end_st_1].to_numpy()
    data_dict['x_st_2'] = df.iloc[:, end_st_1:end_st_2].to_numpy()
    data_dict['x_st_3'] = df.iloc[:, end_st_2:end_st_3].to_numpy()
    data_dict['y_st_1'] = df.iloc[:, end_st_3:end_y1].to_numpy()
    data_dict['y_st_2'] = df.iloc[:, end_y1:].to_numpy()
    
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
    
  elif form == 'tabular':
    features = df.iloc[:, :end_st_3].to_numpy()
    labels = df.iloc[:, end_st_3:].to_numpy()

    return features, labels
    
  elif form == 'dataframe':
    features = df.iloc[:, :end_st_3]
    labels = df.iloc[:, end_st_3:]

    return features, labels
  
  
def convert_wf(df: pd.DataFrame, form: str) -> (dict[str, np.ndarray] | 
  tuple[np.ndarray, np.ndarray] | tuple[pd.DataFrame, pd.DataFrame]):
  """
  """
  
  # set values from config file
  hist_window = 288
  pred_window = 288
  n_times = 3
  n_states = 10
  n_data = len(df.index)
  
  # set starting and end indices of tabular features
  end_s = 2
  end_t1 = end_s + hist_window * n_times
  end_st = end_t1 + hist_window * n_states
  end_t2 = end_st + pred_window * n_times
  
  if form == 'uniform':
    data_dict = {}
    data_dict['x_s'] = df.iloc[:, :end_s].to_numpy()
    data_dict['x_t_1'] = df.iloc[:, end_s:end_t1].to_numpy().astype(int)
    data_dict['x_st'] = df.iloc[:, end_t1:end_st].to_numpy()
    data_dict['x_t_2'] = df.iloc[:, end_st:end_t2].to_numpy().astype(int)
    data_dict['y_st'] = df.iloc[:, end_t2:].to_numpy()
    
    # either order='C' with shape (n_data, n_states, hist_window)
    # or order='F' with shape (n_data, hist_window, n_states)
    data_dict['x_t_1'] = np.reshape(data_dict['x_t_1'], 
      (n_data, n_times, hist_window), order='C')
    data_dict['x_t_2'] = np.reshape(data_dict['x_t_2'], 
      (n_data, n_times, pred_window), order='C')
    data_dict['x_st'] = np.reshape(data_dict['x_st'], 
      (n_data, n_states, hist_window), order='C')
      
    return data_dict
    
  elif form == 'tabular':
    features = df.iloc[:, :end_t2].to_numpy()
    labels = df.iloc[:, end_t2:].to_numpy()

    return features, labels
    
  elif form == 'dataframe':
    features = df.iloc[:, :end_t2]
    labels = df.iloc[:, end_t2:]

    return features, labels
  
  
def convert_um(df: pd.DataFrame, form: str) -> dict:
  """
  """
  
  end_s1 = 1
  end_s2 = end_s1 + 6
  end_t = end_s2 + 4
  
  if form == 'uniform':
    data_dict = {}
    data_dict['x_s_1'] = df.iloc[:, :end_s1].to_numpy()
    data_dict['x_s_2'] = df.iloc[:, end_s1:end_s2].to_numpy()
    data_dict['x_t'] = df.iloc[:, end_s2:end_t].to_numpy()
    data_dict['y_st'] = df.iloc[:, end_t:].to_numpy()
    
    return data_dict
    
  elif form == 'tabular':
    features = df.iloc[:, :end_t].to_numpy()
    labels = df.iloc[:, end_t:].to_numpy()
    
    return features, labels
    
  elif form == 'dataframe':
    features = df.iloc[:, :end_t]
    labels = df.iloc[:, end_t:]

    return features, labels
  
  
def convert_be(df: pd.DataFrame, form: str) -> (dict[str, np.ndarray] | 
  tuple[np.ndarray, np.ndarray] | tuple[pd.DataFrame, pd.DataFrame]):
  """
  """
  
  # set values from config file
  hist_window = 24
  n_states = 9
  n_data = len(df.index)
  
  # set starting and end indices of tabular features
  end_t = 5
  start_st = end_t + 1
  end_st = start_st + hist_window * n_states
  
  if form == 'uniform':
    data_dict = {}
    data_dict['x_t'] = df.iloc[:, :end_t].to_numpy().astype(int)
    data_dict['x_s'] = df.iloc[:, end_t].to_numpy().astype(int)
    data_dict['x_st'] = df.iloc[:, start_st:end_st].to_numpy()
    data_dict['y_st'] = df.iloc[:, end_st:].to_numpy()
    
    # either order='C' with shape (n_data, n_states, hist_window)
    # or order='F' with shape (n_data, hist_window, n_states)
    data_dict['x_st'] = np.reshape(data_dict['x_st'], 
      (n_data, n_states, hist_window), order='C')
    
    return data_dict
    
  elif form == 'tabular':
    features = df.iloc[:, :end_st].to_numpy()
    labels = df.iloc[:, end_st:].to_numpy()

    return features, labels
    
  elif form == 'dataframe':
    features = df.iloc[:, :end_st]
    labels = df.iloc[:, end_st:]
    
    return features, labels
    
    
def load_add_be(path_to_data_subtask: str, form: str):
  """
  """
  
  # set path and load
  path = path_to_data_subtask + 'additional/id_histo_map.csv'
  x_s = pd.read_csv(path)
  
  # transform is chosen such
  if form == 'uniform' or form == 'tabular':
    x_s = x_s.to_numpy()
    
  # set as dictionary and return value
  add = {'x_s': x_s}
  
  return add    
    
    
def load_json_fast(dir: str, filenames: list[str]) -> dict:
  """
  """
  
  def load_json(path):
    with open(path, 'r') as json_file:
      data_dict = json.load(json_file)
  
    return data_dict
  
  with ThreadPoolExecutor() as executor:
    futures = [executor.submit(load_json, dir + fname) for fname in filenames]
    dict_concantenated = {}
    for f in tqdm(futures):
      dict_concantenated.update(f.result())
      
  return dict_concantenated
  
  

def load_csv_fast(dir: str, filenames: list[str]) -> pd.DataFrame:
  """
  """
  
  def load_csv(path):
    return pd.read_csv(path, dtype=np.float32)

  with ProcessPoolExecutor() as executor:
    futures = [executor.submit(load_csv, dir + fname) for fname in filenames]
    dfs = []
    for f in tqdm(futures):
      dfs.append(f.result())

      
  df_concatenated = pd.concat(dfs, ignore_index=True, copy=False)
  
  return df_concatenated
  
  
  
  
