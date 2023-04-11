import pandas as pd
import os
import numpy as np
import gc
from tqdm import tqdm
import math

from load_config import config_BE

def process_all_datasets(config: dict):
  """
  Processes all datasets for Building Electricity task.
  """
  print("Processing Building Electricity dataset.")
  
  # iterated over all subtasks
  for subtask in config['BuildingElectricity']['subtask_list']:
    # augment config with currently iterated subtask paths
    config_building = config_BE(config, subtask)
    
    # import all data
    df_consumption, df_building_images, df_meteo_dict = import_all_data(
      config_building)
      
    # change building IDs here
    df_consumption, df_building_images = adjust_building_ids(
      df_consumption, df_building_images)
      
    # process building imagery
    df_building_images = save_building_imagery(config_building, 
      df_building_images)
    
    # free up memory
    del df_building_images
    gc.collect()
    
    # process meteo data and load profiles
    df_dataset = process_meteo_and_load_profiles(config_building, 
      df_consumption, df_meteo_dict)
      
    # free up memory
    del df_consumption, df_meteo_dict
    gc.collect()
    
    # Do trainining, validation and testing split
    split_train_val_test(config_building, df_dataset)
    
    # free up memory
    del df_dataset
    gc.collect()

    
def import_all_data(config_building: dict) -> (pd.DataFrame, pd.DataFrame, 
  pd.DataFrame):
  """
  Imports electric consumption profiles profiles, building imagery pixel data,
  and meteorological data.
  """
  # import all electric consumption profiles
  df_consumption = pd.read_csv(
    config_building['path_to_raw_building_year_profiles_file'])
    
  # import image pixel histogram values
  df_building_images = pd.read_csv(
    config_building['path_to_raw_aerial_imagery_file'])
    
  # create path to sample meteo files
  meteo_filename_list = os.listdir(
    config_building['path_to_raw_meteo_data_folder'])
    
  # decleare empty dictionary for saving all meteo dataframes
  df_meteo_dict = {}
  
  # iterate over all filenames
  for filename in meteo_filename_list:
    # create full path to iterated file
    path_to_meteo_file = (config_building['path_to_raw_meteo_data_folder'] 
      + filename)
      
    # import meteorological data
    df_meteo = pd.read_csv(path_to_meteo_file)
    
    # save imported dataframe to dataframe dictionary
    df_meteo_dict[filename] = df_meteo
    
  return df_consumption, df_building_images, df_meteo_dict
  

def adjust_building_ids(df_consumption: pd.DataFrame, 
  df_building_images: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
  """
  Maps original building IDs into a new set of IDs starting from 1 to length
  """
  # create a list of all available building IDs
  building_id_list = list(df_consumption.columns.values[1:])
  
  # create empty dict for mapping old to new building IDs from zero to length
  build_id_map_dict = {}
  new_col_list = []
  for count_index, building_id_old in enumerate(building_id_list):
    building_id_new = count_index + 1
    build_id_map_dict[building_id_old] = building_id_new
    new_col_list.append(building_id_new)
    
  # do the renaming
  df_consumption.rename(columns=build_id_map_dict, inplace=True)
  df_building_images.rename(columns=build_id_map_dict, inplace=True)
  
  # shorten up to renaming maximum
  df_building_images = df_building_images[new_col_list]
  new_col_list.insert('building ID', 0)
  df_consumption = df_consumption[new_col_list]
  
  return df_consumption, df_building_images
    

def save_building_imagery(config_building: dict, 
  df_building_images: pd.DataFrame) -> (pd.DataFrame):
  """
  Sorts after IDs, and saves and returns file with new order.
  """
  # get list of columns
  columns_df_list = list(df_building_images.columns.values.astype(int))
  
  # sort in ascending order
  columns_df_list.sort()
  
  # turn back into string
  #columns_df_list = map(str, columns_df_list)
  
  # rearrange df_building_images only
  df_building_images = df_building_images[columns_df_list]
  
  # create saving path for building imagery
  saving_path = config_building['path_to_data_add']+ 'id_histo_map.csv'
  
  # save df_building_images_new
  df_building_images.to_csv(saving_path, index=False)
  
  return df_building_images

    
def process_meteo_and_load_profiles(config_building: dict, 
  df_consumption: pd.DataFrame, df_meteo_dict: pd.DataFrame) -> pd.DataFrame:
  """
  Main data processing. Takes electric load profiles and meteorological data
  and combines these into a single dataset dataframe.
  """
  # Create dataframe column
  new_df_columns_base = ['year', 'month', 'day', 'hour', 'quarter_hour', 
    'building_id']
    
  # fill a separate list 
  new_df_columns = new_df_columns_base.copy()
  
  # append column entries for meteorological data
  for column_name in config_building['meteo_name_list']:
    for pred_time_step in range(config_building['historic_window']):
      entry_name = '{}_{}'.format(column_name, pred_time_step+1)
      new_df_columns.append(entry_name)
      
  # append column entries for electric load
  for pred_time_step in range(config_building['prediction_window']):
    entry_name = 'load_{}'.format(pred_time_step+1)
    new_df_columns.append(entry_name)
      
  # Create datasets
  df_consumption.drop(index=1, inplace=True)
  
  # get corresponding time stamps series and reset indices
  time_stamps = df_consumption['building ID'].iloc[1:].reset_index(drop=True)
  
  # create a list of all building IDs
  building_id_list = list(df_consumption.columns.values[1:])
  
  # decleare empty values array. Filling matrix pre-allocates memory and
  # decreases computational time significantly.
  values_array = np.zeros((len(building_id_list) * (
      len(time_stamps) - config_building['historic_window'] 
        - config_building['prediction_window']),
    (len(new_df_columns_base) + config_building['historic_window'] * (
      len(config_building['meteo_name_list'])) 
      + config_building['prediction_window'])))
      
  # create progress bar
  pbar = tqdm(total=len(building_id_list))
  
  # declare df row counter
  datapoint_counter = 0
  
  # iterate over all building IDs
  for building_id in building_id_list:
    # get cluster id as integer
    cluster_id = df_consumption[building_id].iloc[0].astype(int)
    
    # get building load with new indices
    building_load = df_consumption[building_id].iloc[1:].reset_index(
      drop=True)
      
    # transform building id into integer
    building_id = int(building_id)
    
    # create key to corresponding meteo data
    key_meteo = 'meteo_{}_2014.csv'.format(cluster_id)
    
    # get corresponding meteorological data
    df_meteo = df_meteo_dict[key_meteo]
    
    # drop local_time column 
    df_meteo = df_meteo.drop(columns=['local_time'])
    
    # iterate over all time stamps in prediction window steps
    for i in range(config_building['historic_window'] * 4, 
      len(time_stamps) - config_building['prediction_window']):
      # get time stamp
      time = time_stamps[i]
      
      # get single entries of timestamp
      year = int(time[0:4])
      month = int(time[5:7])
      day = int(time[8:10])
      hour = int(time[11:13])
      quarter_hour = int(time[14:16])
      
      # get iterated meteorological data
      meteo_dict = {}
      for meteo_name in config_building['meteo_name_list']:
        meteo_values = df_meteo[meteo_name].iloc[
          list(range(i-config_building['historic_window'] *4, i, 4))
        ].values
        meteo_dict[meteo_name] = meteo_values
        
      # get iterated load profile data
      load_profile = building_load[
        i:(i+config_building['prediction_window'])].values
        
      # add features to values_array. Ensures same order as new_df_columns.
      for index_col, entry_name in enumerate(new_df_columns_base):
        # build the command
        command = 'values_array[datapoint_counter,index_col]={}'.format(
          entry_name)
          
        # execute the command
        exec(command)
        
      # add meteorological data to entry
      for meteo_name, meteo_profile in meteo_dict.items():
        for i in range(len(meteo_profile)):
          index_col += 1
          values_array[datapoint_counter,index_col] = meteo_profile[i]
          
      # add load profile to entry
      for i in range(len(load_profile)):
        index_col += 1
        values_array[datapoint_counter, index_col] = load_profile[i]
        
      # increment df row counter
      datapoint_counter += 1
      
    # increment progbar
    pbar.update(1) 
    
  # create dataframe from filled matrix values
  df_dataset = pd.DataFrame(data=values_array, columns=new_df_columns)
  
  # drop zero entries
  df_dataset = df_dataset.loc[~(df_dataset==0).all(axis=1)]
  
  # free up memory
  del values_array
  gc.collect()
  
  return df_dataset
    
    
def split_train_val_test(config_building: dict, df_dataset: pd.DataFrame):
  """
  Splits and saves datasets according to configuration rules.
  """
  # get total number of data points 
  n_data_total = len(df_dataset)
  
  # get the out-of-distribution splitting rules
  temporal_ood = config_building['temporal_ood']
  spatial_ood = config_building['spatial_ood']
  
  # create spatial ood split
  df_testing = df_dataset.loc[
    (df_dataset['building_id'].isin(spatial_ood['ood_building_ids']))]
    
  # remove split spatial ood data points
  df_dataset = df_dataset.drop(df_testing.index)
  
  # create temporal ood split
  df_temporal_ood = df_dataset.loc[
    (df_dataset['month'].isin(temporal_ood['ood_months']))
    | (df_dataset['day'].isin(temporal_ood['ood_days']))
    | (df_dataset['hour'].isin(temporal_ood['ood_hours']))
    | (df_dataset['quarter_hour'].isin(temporal_ood['ood_quarter_hours']))]
    
  # remove the spatial ood split data points which is training dataset
  df_training = df_dataset.drop(df_temporal_ood.index)
  
  # free up memory
  del df_dataset
  gc.collect()
  
  # append to testing dataset
  df_testing = pd.concat([df_testing, df_temporal_ood], ignore_index=True)
  
  # free up memory
  del df_temporal_ood
  gc.collect()
  
  # do validation split
  df_validation = df_testing.sample(
    frac=config_building['val_test_split'], 
    random_state=config_building['seed'])
    
  # remove validation data split from testing dataset
  df_testing = df_testing.drop(df_validation.index)
  
  # calculate and analyze dataset properties
  n_train, n_val, n_test = len(df_training), len(df_validation), len(df_testing)
  n_total = n_train + n_val + n_test
  
  # print
  print("Training data   :   {}/{} {:.0%}".format(n_train, n_total, 
      n_train/n_total),
    "\nValidation data :   {}/{} {:.0%}".format(n_val, n_total,
      n_val/n_total),
    "\nTesting data    :   {}/{} {:.0%}".format(n_test, n_total,
      n_test/n_total))
      
  # test if all indexes dropped correctly.
  if n_data_total != n_total:
      print("Error! Number of available data is {}".format(n_data_total),
          "and does not match number of resulting data {}.".format(n_total))
          
  # save results in chunks
  save_in_chunks(config_building,
    config_building['path_to_data_train'] + 'training_data', df_training)
  save_in_chunks(config_building,
    config_building['path_to_data_val'] + 'validation_data', df_validation)
  save_in_chunks(config_building,
    config_building['path_to_data_test'] + 'testing_data', df_testing)

    
def save_in_chunks(config_building: dict, saving_path: str, df: pd.DataFrame):
  """
  Shuffles dataframe, then saves it in chunks with number of datapoints per 
  file defined by config such that each file takes less than about 1 GB size.
  """
  df = df.sample(frac=1, random_state=config_building['seed'])
  
  for file_counter in range(1, 312321321312):
    path_to_saving = saving_path + '_{}.csv'.format(file_counter)
    
    df.iloc[:config_building['data_per_file']].to_csv(
      path_to_saving, index=False)
      
    df = df[config_building['data_per_file']:]
    
    if len(df) == 0:
      break
            
            
