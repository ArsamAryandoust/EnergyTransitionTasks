import selberai.data.load_data as load_data

import numpy as np
def test(config: dict, name: str, subtask: str):
  """
  """
  print('Testing {} for subtask {}'.format(name, subtask))
  
  ###
  # Load data
  ###
  
  # set the path to data and token we want to load
  path_to_data = config['general']['path_to_data']   
  path_to_token = config['dataverse']['path_to_token']
  
  # load data
  dataset = load_data.load(name, subtask, path_to_data=path_to_data)
  
  
  ###
  # Test variance of features and labels
  ###
  
  var_train_x_t = np.var(dataset.train['x_t'], axis=0)
  var_train_x_s = np.var(dataset.train['x_s'], axis=0)
  var_train_x_st = np.var(dataset.train['x_st'], axis=0)
  var_train_y = np.var(dataset.train['y'], axis=0)
  
  var_val_x_t = np.var(dataset.val['x_t'], axis=0)
  var_val_x_s = np.var(dataset.val['x_s'], axis=0)
  var_val_x_st = np.var(dataset.val['x_st'], axis=0)
  var_val_y = np.var(dataset.val['y'], axis=0)
  
  var_test_x_t = np.var(dataset.test['x_t'], axis=0)
  var_test_x_s = np.var(dataset.test['x_s'], axis=0)
  var_test_x_st = np.var(dataset.test['x_st'], axis=0)
  var_test_y = np.var(dataset.test['y'], axis=0)
  
  print('Variance train x_st:', var_train_x_st)
  print('Variance val x_st:', var_val_x_st)
  print('Variance test x_st:', var_test_x_st)
