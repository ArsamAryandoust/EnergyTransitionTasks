import os
import gc

import pandas as pd
import numpy as np
from tqdm import tqdm

from config.load_wf import config_WF

def process_all_datasets(config: dict, save: bool):
  """
  Does the main data processing.
  """
  print("\nProcessing Wind Farm dataset.")
  
  for subtask in config['WindFarm']['subtask_list']:
    # augment configuration with additional information
    config_wind = config_WF(config, subtask, save)
    
    # load data of this subtask
    df_data, df_locations = import_all_data(config_wind)
    
    # expand timestamp
    df_data = expand_timestamp(df_data)
    
    # clean data
    df_data = clean_data(df_data)
    
    # create data points with sliding time window
    df_data = create_datapoints(config_wind, df_data)
    
    # split the loaded dataframe into training, validation and testing
    split_train_val_test(config_wind, df_data, df_locations, save)
        
        
def import_all_data(config_wind: dict) -> (pd.DataFrame, pd.DataFrame):
  """
  Loads the SCADA data and the location data of each turbine from csv files,
  and returns these.
  """    
  print('\nLoading data for {}.'.format(config_wind['subtask']))
  
  # load turbine location dataframe
  df_locations = pd.read_csv(config_wind['path_to_turb_loc_file'])
  
  # for days_245, all data is in single file
  if config_wind['subtask'] == 'days_245':
    df_data = pd.read_csv(config_wind['path_to_data_raw_file'])
  
  # for days_177, data is in multiple files
  elif config_wind['subtask'] == 'days_177':
    # get list of filenames for input and output of challenge
    list_of_files_in = os.listdir(
      config_wind['path_to_data_raw_infile_folder'])
    list_of_files_out = os.listdir(
      config_wind['path_to_data_raw_outfile_folder'])
    
    # declare empty dataframe data
    df_data = pd.DataFrame()                 
    
    # create progress bar
    pbar = tqdm(total=len(list_of_files_in))
    
    # iterate over both folder files simultaneously
    for in_filename, out_filename in zip(list_of_files_in, 
      list_of_files_out):
      
      # set paths and load data
      path = config_wind['path_to_data_raw_infile_folder'] + in_filename
      df_infile = pd.read_csv(path)
      path = config_wind['path_to_data_raw_outfile_folder'] + out_filename
      df_outfile = pd.read_csv(path)
      
      # concatenate dataframes
      df_data = pd.concat([df_data, df_infile, df_outfile], 
        ignore_index=True)
        
      # data comes in sliding window, so drop the duplicates
      df_data.drop_duplicates(inplace=True, ignore_index=True)
      
      #update progress bar
      pbar.update(1)
          
  # subsample/shuffle
  df_data = df_data.sample(frac=config_wind['subsample_frac'], 
    random_state=config_wind['seed'])
    
  return df_data, df_locations
    
    
def expand_timestamp(df_data: pd.DataFrame) -> (pd.DataFrame):
  """
  """
  # split timestamp column into separate columns with hour and minute
  df_data[['hour', 'minute']] = df_data.Tmstamp.str.split(':', expand=True)
  
  # drop the original timestamp column
  df_data.drop(columns=['Tmstamp'], inplace=True)
  
  # Put hour and minute columns next to Day column. NOTE: not necessary, but 
  # reads better in notebooks
  cols = df_data.columns.tolist()
  cols = cols[:2] + cols[-2:] + cols[2:-2]
  df_data = df_data[cols]
  
  return df_data
  
  
def clean_data(df_data: pd.DataFrame) -> (pd.DataFrame):
  """
  """
  # get number of originally available raw data points
  n_0 = len(df_data.index)
  
  # measurement error
  df_data.drop(df_data[(df_data.Patv <= 0) & (df_data.Wspd >= 2.5)].index,
    inplace=True)
    
  # set negative generation to zero
  index_list = df_data[(df_data.Patv < 0)].index.to_list()
  df_data.loc[index_list, 'Patv'] = 0
  df_data.loc[index_list, 'Prtv'] = 0
  
  # measurement error
  df_data.drop(df_data[
    (df_data.Pab1 > 89) | (df_data.Pab2 > 89) | (df_data.Pab3 > 89)].index,
    inplace=True)

  # anomaly
  df_data.drop(df_data[(df_data.Ndir < -720) | (df_data.Ndir > 720)].index,
    inplace=True)
  
  # anomaly 
  df_data.drop(df_data[(df_data.Wdir < -180) | (df_data.Wdir > 180)].index,
    inplace=True)
  
  # drop NaN rows
  df_data.drop(df_data[df_data.isnull().any(axis=1)].index,
    inplace=True)
    
  
  # get number of raw data points after cleaning operations
  n_1 = len(df_data.index)
  
  # tell us how much we cleaned
  print(
    'During cleaning process {} ({:.0%}) raw data points were removed'.format(
    n_0-n_1, 1-n_1/n_0))
  
  return df_data
  
    
def create_datapoints(config_wind: dict, df_data: pd.DataFrame) -> pd.DataFrame:
  """
  """
  print('\nCreating data points for {}!'.format(config_wind['subtask']))
  
  # get a list of all turbine IDs available in data
  turbine_list = list(set(df_data['TurbID']))
  
  # set number of maximum days
  n_days = len(set(df_data['Day']))
  
  # set number of rows
  n_rows = (len(turbine_list) * n_days * 24 * 6 - config_wind['historic_window'] 
    - config_wind['prediction_window'])
  
  # set number of columns
  n_cols = (1 + 3 * config_wind['historic_window'] 
    + len(config_wind['fseries_name_list']) * config_wind['historic_window']
    + 3 * config_wind['prediction_window']
    + 1 * config_wind['prediction_window'])
  
  # create zero values array in maximum size it can fill given no sparsity
  values_array = np.zeros((n_rows, n_cols)).astype('f8')
    
  # set a datapoint counter
  data_counter = 0
  
  # create progress bar
  pbar = tqdm(total=len(turbine_list))
  
  # iterate over all turbine IDs
  for turbine_id in turbine_list:
  
    # get corresponding entries
    df_turbine = df_data.loc[df_data['TurbID']==turbine_id].copy()
    
    # sort by time
    df_turbine.sort_values(by=['Day', 'hour', 'minute'], inplace=True,
      ignore_index=True)
      
    
    # iterate over entries of df_turbine
    for i in range(config_wind['historic_window'], 
      len(df_turbine) - config_wind['prediction_window']):
      
      # set column counter
      col_counter = 0
      
      # set spatial and temporal values
      values_array[data_counter, col_counter] = turbine_id
      # increment column counter
      col_counter =+ 1
      
      # iterate over historic time window to add first time-variant features
      for j in range(i, i-config_wind['historic_window'], -1):
        
        # add time-variant features      
        values_array[data_counter, col_counter] = df_turbine['Day'][j]
        values_array[data_counter, col_counter+1] = df_turbine['hour'][j]
        values_array[data_counter, col_counter+2] = df_turbine['minute'][j]
        
        # increment column counter
        col_counter += 3
        
      # iterate over historic time window to add space-time-variant features
      for j in range(i, i-config_wind['historic_window'], -1):
      
        # iterate over time series feature names
        for colname in config_wind['fseries_name_list']:
          
          # add space-time-variant features
          values_array[data_counter, col_counter] = df_turbine[colname][j]
          
          # increment column counter
          col_counter += 1
          
      # iterate over prediction window to add second time-variant features
      for j in range(i, i+config_wind['prediction_window']):
        
        # add time-variant features      
        values_array[data_counter, col_counter] = df_turbine['Day'][j]
        values_array[data_counter, col_counter+1] = df_turbine['hour'][j]
        values_array[data_counter, col_counter+2] = df_turbine['minute'][j]
        
        # increment column counter
        col_counter += 3
            
      # iterate over prediction window to add labels
      for j in range(i, i+config_wind['prediction_window']):
      
        # set active power as label to be predicted
        values_array[data_counter, col_counter] = df_turbine['Patv'][j]
        
        # increment column counter
        col_counter += 1
        
      # increment counter
      data_counter += 1
      
    # update progress bar for turbine count
    pbar.update(1)
  
  ### Create column name list ###
  col_name_list = ['TurbID']
  time_stamp_list = ['day', 'hour', 'minute']
  new_fseries_name_list = ['wind_speed', 'wind_direction', 'temperature_out',
  'temperature_in', 'nacelle_angle', 'blade1_angle', 'blade2_angle',
  'blade3_angle', 'reactive_power', 'active_power']
  
  # add first time-variant column names
  for i in range(1, config_wind['historic_window']+1):
    for colname_base in time_stamp_list:
      colname = 'hist_{}_{}'.format(colname_base, i)
      col_name_list.append(colname)
  
  # add space-time-variant column names
  for i in range(1, config_wind['historic_window']+1):
    for colname_base in new_fseries_name_list:
      colname = '{}_{}'.format(colname_base, i)
      col_name_list.append(colname)
      
  # add second time-variant column names
  for i in range(1, config_wind['prediction_window']+1):
    for colname_base in time_stamp_list:
      colname = 'pred_{}_{}'.format(colname_base, i)
      col_name_list.append(colname)
      
  # add future data columns
  for i in range(1, config_wind['prediction_window']+1):
    colname = 'pred_active_power_{}'.format(i)
    col_name_list.append(colname)
  
  # create dataframe and overwrite old one
  df_data = pd.DataFrame(values_array, columns=col_name_list)   
     
  # get number of data points before dropping zero entries
  n_0 = len(df_data.index)
  
  # drop zero entries
  df_data = df_data.loc[~(df_data==0).all(axis=1)]
  
  # get number of data points after dropping zero entries
  n_1 = len(df_data.index)
  
  # tell us how much got dropped
  print('\n{} data points were dropped ({:.2%} sparsity).'.format(
    n_0-n_1, (n_0-n_1)/n_0))
    
  return df_data
  
  
def split_train_val_test(config_wind: dict, df_data: pd.DataFrame,
  df_locations: pd.DataFrame, save: bool):
  """
  """
  # get total number of data points 
  n_data_total = len(df_data)
  
  # Split training and ood testing
  temporal_ood = config_wind['temporal_ood']
  
  # split of temporal ood and simply name df_test
  df_test = df_data.loc[
    (df_data['hist_day_1'].isin(temporal_ood['ood_days']))
    | (df_data['hist_hour_1'].isin(temporal_ood['ood_hours']))
    | (df_data['hist_minute_1'].isin(temporal_ood['ood_minutes']))]
    
  # drop separated indices
  df_data = df_data.drop(df_test.index)
  
  # get splitting rules
  spatial_ood = config_wind['spatial_ood']
  
  # split of temporal ood
  df_spatial_test = df_data.loc[
    df_data['TurbID'].isin(spatial_ood['ood_turbine_ids'])]
    
  # drop separated indices
  df_data = df_data.drop(df_spatial_test.index)
  
  # concat to test
  df_test = pd.concat([df_test, df_spatial_test], ignore_index=True)
  
  # free up memory
  del df_spatial_test
  gc.collect()
  
  # Augment dataframes with location data by mapping to ID and drop ID
  df_data = pd.merge(df_data, df_locations, on='TurbID', how='left')
  df_test = pd.merge(df_test, df_locations, on='TurbID', how='left')
  df_data.drop(columns=['TurbID'], inplace=True)
  df_test.drop(columns=['TurbID'], inplace=True)
  
  # rearrange columns: put x and y positions first 
  col_list = df_data.columns.to_list()
  col_list.remove('x')
  col_list.remove('y')
  col_list = ['x', 'y'] + col_list
  df_data = df_data[col_list]
  df_test = df_test[col_list]
  
  # free up memory
  del df_locations
  gc.collect()
  
  # Split ood validation and testing
  df_val = df_test.sample(frac=config_wind['val_test_split'], 
    random_state=config_wind['seed'])
    
  # remove validation data split from testing dataset
  df_test.drop(df_val.index, inplace=True)
  
  # Calculate and analyze dataset properties
  n_train, n_val, n_test = len(df_data), len(df_val), len(df_test)
  n_total = n_train + n_val + n_test
  print("Training data   :   {}/{} {:.0%}".format(n_train, n_total, 
      n_train/n_total),
    "\nValidation data :   {}/{} {:.0%}".format(n_val, n_total,
      n_val/n_total),
    "\nTesting data    :   {}/{} {:.0%}".format(n_test, n_total,
      n_test/n_total))   
      
  # small test if all indexes dropped correctly.
  if n_data_total != n_total:
    print("Error! Number of available data is {}".format(n_data_total),
      "and does not match number of resulting data {}.".format(n_total))
      
  # save results in chunks
  save_in_chunks(config_wind,
    config_wind['path_to_data_train'] + 'training_data', df_data, save)
  save_in_chunks(config_wind,
    config_wind['path_to_data_val'] + 'validation_data', df_val, save)
  save_in_chunks(config_wind,
    config_wind['path_to_data_test'] + 'testing_data', df_test, save)
  
  
def save_in_chunks(config_wind: dict, saving_path: str, df: pd.DataFrame,
  save: bool):
  """
  Shuffles dataframe, then saves it in chunks with number of datapoints per 
  file defined by config such that each file takes less than about 1 GB size.
  """
  # shuffle dataframe
  df = df.sample(frac=1, random_state=config_wind['seed'], ignore_index=True)
  
  for file_counter in range(1, 312321321312):
    if len(df) == 0:
      break 
      
    if save:
      # set saving path
      path_to_saving = saving_path + '_{}.csv'.format(file_counter)
      
      # save as csv
      df.iloc[:config_wind['data_per_file']].to_csv(
        path_to_saving, index=False)
      
    # shorten dataframe
    df = df[config_wind['data_per_file']:]
  
  
  
  
