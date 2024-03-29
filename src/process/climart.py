import os
import gc
import h5py
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.load_ca import config_CA


def process_all_datasets(config: dict, save: bool):
  """
  Loads ClimART datasets, processes them and saves resulting datasets. At 
  the time of processing the original data, two years have been containing
  errorneous files, that is 1995 (inputs) and 1851 (outputs).
  """
  print("Processing ClimArt dataset.")
  
  # iterate over all subtasks
  for subtask in config['ClimART']['subtask_list']:
    # augment conigurations with additional information
    config_climart = config_CA(config, subtask, save=save)
    
    # do all data processing
    process_split_all_data(config_climart, save=save)
    
        
def process_split_all_data(config_climart: dict, save: bool):
  """
  Main function doing all the data processing for ClimART. At the time of
  processing the original data, two years have been containing errorneous 
  files, that is 1995 (inputs) and 1851 (outputs).
  """
  # list all data available in one of the raw data foulders.
  list_of_years = os.listdir(config_climart['path_to_data_raw_inputs'])
  
  # remove file ending from list of years
  list_of_years = [list_entry[:-3] for list_entry in list_of_years]
  
  # import meta data
  feature_by_var = import_meta_json(config_climart)
  
  # decleare empty dataframes for trainining validation and testing
  df_train = pd.DataFrame()
  df_val = pd.DataFrame()
  df_test= pd.DataFrame()
  
  # declare data point counters
  train_chunk_counter, val_chunk_counter, test_chunk_counter = 0, 0, 0
  
  # Files for these files are corrupted
  #list_of_years.remove('1995') # inputs
  #list_of_years.remove('1851') # outputs pristine
  
  # create progress bar
  pbar = tqdm(total=len(list_of_years))
  
  # iterate over all available years
  for year in list_of_years:
    
    # import inputs and outputs
    inputs, outputs = import_h5_data(config_climart, year)
    
    # process raw data
    df_inputs, df_outputs = process_raw_data(config_climart, feature_by_var, 
      inputs, outputs)
        
    # free up memory
    inputs.close()
    outputs.close()
    del inputs, outputs
    gc.collect()
    
    # augment and merge data
    df = augment_and_merge(year, df_inputs, df_outputs)   
    
    # free up memory
    del df_inputs, df_outputs 
    gc.collect()
    
    # clean data
    df = clean_data(df)
    
    # subsample data
    df = df.sample(frac=config_climart['subsample_frac'],
      random_state=config_climart['seed'])
        
    # check if this is a testing year data
    if config_climart['temporal_ood']['year'] == int(year):
      # append entire datasets to test dataframes
      df_test = pd.concat([df_test, df], ignore_index=True)
      
      # free up memory
      del df 
      gc.collect()
        
    else:
      # extract the rows from dataframes with indices for test coordinates
      df_test_coordiantes = df[
        df.index.isin(config_climart['spatial_ood']['coordinates'])]
          
      # append extracted rows to test dataframes
      df_test = pd.concat([df_test, df_test_coordiantes], ignore_index=True)
      
      # drop
      df.drop(df_test_coordiantes.index, inplace=True)
      
      # free up memory
      del df_test_coordiantes
      gc.collect()
      
      # extract the rows from dataframes with matching hours of year
      df_test_hours_of_year = df.loc[df['hour_of_year'].isin(
        config_climart['temporal_ood']['hours_of_year'])]
              
      # append extracted rows to test dataframes
      df_test = pd.concat([df_test, df_test_hours_of_year], ignore_index=True)
      
      # drop
      df = df.drop(df_test_hours_of_year.index)
      
      # free up memory
      del df_test_hours_of_year
      gc.collect()
      
      # append training dataset
      df_train = pd.concat([df_train, df], ignore_index=True)
      
      # free up memory     
      del df
      gc.collect()
      
    # this condition guarantees validation splits at good moments
    if (len(df_test) * (1 - config_climart['val_test_split'])
      > config_climart['data_per_file']):
      
      # create training and validation datasets
      df_val_append = df_test.sample(
        frac=config_climart['val_test_split'],
        random_state=config_climart['seed'])
      df_test.drop(df_val_append.index, inplace=True)
      
      # append validation dataset
      df_val = pd.concat([df_val, df_val_append], ignore_index=True)
      
      # free up memory     
      del df_val_append
      gc.collect()
      
      # save resulting data in chunks
      df_val, val_chunk_counter = save_chunk(config_climart, df_val, 
        val_chunk_counter, config_climart['path_to_data_subtask_val'], 
        'validation', save=save)
      df_test, test_chunk_counter = save_chunk(config_climart, df_test,
        test_chunk_counter, config_climart['path_to_data_subtask_test'],
        'testing', save=save)
    
    # save resulting data in chunks
    df_train, train_chunk_counter = save_chunk(config_climart, df_train, 
      train_chunk_counter, config_climart['path_to_data_subtask_train'], 
      'training', save=save)
        
    # update progbar
    pbar.update(1)
    
      
  ### Tell us the ratios that result from our splitting rules
  n_train = (train_chunk_counter * config_climart['data_per_file'] 
    + len(df_train))
  n_val = (val_chunk_counter * config_climart['data_per_file'] 
    + len(df_val))
  n_test = (test_chunk_counter * config_climart['data_per_file'] 
    + len(df_test))
  n_total = n_train + n_val + n_test
  
  # print out
  print("Training data   :   {}/{} {:.0%}".format(n_train, n_total, 
    n_train/n_total),
    "\nValidation data :   {}/{} {:.0%}".format(n_val, n_total,
    n_val/n_total),
    "\nTesting data    :   {}/{} {:.0%}".format(n_test, n_total,
    n_test/n_total))
  
  ### Save results of last iteration
  df_train, train_chunk_counter = save_chunk(config_climart, df_train, 
    train_chunk_counter, config_climart['path_to_data_subtask_train'], 
    'training', save=save, last_iteration=True)
  df_val, val_chunk_counter = save_chunk(config_climart, df_val, 
    val_chunk_counter, config_climart['path_to_data_subtask_val'], 
    'validation', save=save, last_iteration=True)
  df_test, test_chunk_counter = save_chunk(config_climart, df_test,
    test_chunk_counter, config_climart['path_to_data_subtask_test'],
    'testing', save=save, last_iteration=True)
    
    
def import_meta_json(config_climart: dict) -> dict:
  """
  Loads the meta data from a json file. The dictionary it returns and we 
  consecutively use contains information about features and labels, and their
  starting and ending indices in the multidimensional array distinguished by 
  'globals', 'levels' and 'layers' values.
  """
  # create path to json file
  path_to_meta_json = config_climart['path_to_data_raw'] + 'META_INFO.json'
  
  # load json file
  with open(path_to_meta_json, 'r') as f:
    meta_dict = json.load(f)
   
  # take and return only feature by variable   
  feature_by_var = meta_dict['feature_by_var']
  
  return feature_by_var
    
    
def import_h5_data(config_climart: dict, year: str):
  """
  Imports inputs and sub-task outputs from .h5 files. Note that inputs of the
  respective subtask are extracted later only.
  """
  # create paths to files
  path_to_inputs = config_climart['path_to_data_raw_inputs'] + year + '.h5'
  path_to_outputs = (config_climart['path_to_data_raw_outputs_subtask'] + year 
    + '.h5')
      
  # load data
  inputs = h5py.File(path_to_inputs, 'r')
  outputs = h5py.File(path_to_outputs, 'r')
  
  return inputs, outputs
    
    
def process_raw_data(config_climart: dict, feature_by_var: dict, 
  inputs, outputs) -> (pd.DataFrame, pd.DataFrame):
  """
  Inputs and outputs are turned into reshaped dataframes.
  """
  # define empty dataframes
  df_inputs = pd.DataFrame()
  df_outputs = pd.DataFrame()
  
  ###
  # Process input data ###
  ###
  
  ### Do for globals ###
  # retrieve data and tranform into numpy arrays
  data = np.array(inputs['globals'])
  
  # create column names
  var_dict = feature_by_var['globals']
  col_names_list = create_input_col_name_list(var_dict)
  
  # transform into dataframe
  data = pd.DataFrame(data, columns=col_names_list)
  
  # append to input dataframes
  df_inputs = pd.concat([df_inputs, data], axis=1)
  
  ### Do for layers ###
  # create column names
  var_dict = feature_by_var['layers']
  col_names_list = create_input_col_name_list(var_dict)
  
  # retrieve data and tranform into numpy arrays
  if config_climart['subtask'] == 'pristine':
    data = np.array(inputs['layers'][:, :, :14])
    col_names_list = col_names_list[:14]
  else:
    data = np.array(inputs['layers'])
    
  # reshape data
  data, col_names_list = reshape(data, col_names_list)
  
  # transform into dataframe
  data = pd.DataFrame(data, columns=col_names_list)
  
  # append to input dataframes
  df_inputs = pd.concat([df_inputs, data], axis=1) 
  
  ### Do for levels ###
  # retrieve data and tranform into numpy arrays
  data = np.array(inputs['levels'])
  
  # create column names
  var_dict = feature_by_var['levels']
  col_names_list = create_input_col_name_list(var_dict)
  
  # reshape data
  data, col_names_list = reshape(data, col_names_list)
  
  # transform into dataframe
  data = pd.DataFrame(data, columns=col_names_list)
  
  # append to input dataframes
  df_inputs = pd.concat([df_inputs, data], axis=1)
  
  # free up memory
  del inputs
  gc.collect()
  
  ###
  # Process output data ###
  ###
  
  # iterate over outputs
  for key in outputs:
    # retrieve data and tranform into numpy arrays
    data = np.array(outputs[key])
    
    # create column names
    if config_climart['subtask'] == 'pristine':
      var_dict = feature_by_var['outputs_pristine']
      
    else:
      var_dict = feature_by_var['outputs_clear_sky']
      
    col_names_list_outputs = create_output_col_name_list(key, var_dict)
    
    # transform into dataframe
    data = pd.DataFrame(data, columns=col_names_list_outputs)
    
    # append to input dataframes
    df_outputs = pd.concat([df_outputs, data], axis=1)
  
  return df_inputs, df_outputs
    
    
def create_input_col_name_list(var_dict: dict) -> list:
  """
  Takes a variables dict and iterates over it. Each entry of the dictionary 
  contains a variable range dictionary with starting and ending indices. 
  For the entire range of the variable, a column entry is created to be
  appended to a column name list for use in a pandas dataframe.
  """
  # declare empty column names list
  col_names_list = []
  
  # iterate over variable name dict
  for var_name in var_dict:
  
    # get dict with start and end range of current variable
    var_range_dict = var_dict[var_name]
    
    # get range size
    range_size = var_range_dict['end'] - var_range_dict['start']
    
    # append column name to list if range size is only 1
    if range_size == 1:
      col_names_list.append(var_name)
      
    elif range_size >1:
    
      for feature_iter in range(range_size):
        col_name = var_name + '_{}'.format(feature_iter)
        col_names_list.append(col_name)
        
    else:
      print('Caution. Something went wrong with creating column names')
      
  return col_names_list
    
    
def create_output_col_name_list(var_name: str, var_dict: dict) -> list:
  """
  Takes a variable name string and appends one entry to a column name list for
  the entire range from start to end of that variable with a previx being the
  variable name. Returns a list of column names for being used in an output 
  dataframe.
  """
  
  # get variable range dictionary
  var_range_dict = var_dict[var_name]
  
  # declare empty column names list
  col_names_list = []
  
  # get range size
  range_size = var_range_dict['end'] - var_range_dict['start']
  
  # append column name to list if range size is only 1
  for feature_iter in range(range_size):
    col_name = var_name + '_{}'.format(feature_iter)
    col_names_list.append(col_name)
      
  return col_names_list    
    
    
def reshape(data, col_names_list: list):
  """
  Reshapes multidimensional data array into a tabular format or matrix and 
  expands the passed col_names_list accordingly. Returns the reshaped data
  array and expanded column name list.
  """
  # get number of data points
  n_data = len(data)
  n_steps = data.shape[1]
  n_features = data.shape[2]
  
  # get number of columns for desired reshaping
  n_columns = n_steps * n_features
  
  # reshape with C order
  data = np.reshape(data, (n_data, n_columns), order='C')
  
  # declare new empty column names list
  expanded_col_names_list = []
  
  # expand col_names_list according to reshape with C order
  for steps in range(n_steps):
  
    # iterate over all column names
    for col_name in col_names_list:
    
      # create entry
      entry= 'l_{}_{}'.format(steps, col_name)
      
      # append entry
      expanded_col_names_list.append(entry)
          
  return data, expanded_col_names_list
    
    
def augment_and_merge(year: str, df_inputs: pd.DataFrame, 
  df_outputs: pd.DataFrame) -> pd.DataFrame:
  """
  Augments datapoints of passed dataframes with year and hour of year entries.
  """
  # concatenate dataframes
  df = pd.concat([df_inputs, df_outputs], axis=1)
  
  # calculate for each data point the hour of year
  n_lat, n_lon, n_hours_per_step = 64, 128, 205
  n_points_space = n_lat * n_lon
  hour_of_year = (np.floor(df.index.values / n_points_space) 
    * n_hours_per_step).astype(int)
      
  # augment data with year
  df.insert(0, 'year', int(year))
  
  # augment data with hour of year
  df.insert(1, 'hour_of_year', hour_of_year)
  
  ### put re-arange order of spatial coordinates for visibility ###
  # get column name lists
  cols = df.columns.tolist()
  
  # remove entries from column list
  cols.remove('x_cord'), cols.remove('y_cord'), cols.remove('z_cord')
  
  # insert them again at new spots    
  cols.insert(2,'x_cord'), cols.insert(3,'y_cord'), cols.insert(4,'z_cord')
  
  # set the new column orders
  df = df[cols]
  
  # important to reset indices here to avoid multiple row losses when dropping 
  # by index later
  df.reset_index(drop=True, inplace=True)
  
  return df
    

def clean_data(df: pd.DataFrame) -> (pd.DataFrame):
  """
  """
  # set number of columns
  n_cols = len(df.columns)

  # set threshold we found for anomalies
  threshold = 1e18

  # get boolean series of bad rows
  bad_rows = ((df < threshold) & (df > -threshold)).sum(axis=1) < n_cols

  # get indices of bad rows
  bad_rows_index = bad_rows[bad_rows].index.values

  # drop bad rows from dataframe
  df.drop(bad_rows_index, inplace=True)
  
  return df
  
  
    
def save_chunk(config_climart: dict, df: pd.DataFrame, chunk_counter: int,
  saving_path: str, filename: str, save: bool, last_iteration=False):
  """
  Save a chunk of dataframe in case it exceeds the number of data points per
  file that we want, or in case we are in the last iteration.
  """
  
  ### Save resulting data in chunks
  while len(df.index) > config_climart['data_per_file'] or last_iteration:
    # increment chunk counter 
    chunk_counter += 1
    
    # create path
    path_to_saving = '{}_{}.csv'.format(saving_path + filename, chunk_counter)
    
    # shuffle
    df = df.sample(frac=1, random_state=config_climart['seed'])
    
    # save chunk
    if save:
    
      if len(df) > 0:
        df.iloc[:config_climart['data_per_file']].to_csv(path_to_saving, 
          index=False)
            
    # delete saved chunk
    df = df[config_climart['data_per_file']:]
    
    # Must be set to exit loop on last iteration
    last_iteration = False
      
  return df, chunk_counter 
    
    
    
    
