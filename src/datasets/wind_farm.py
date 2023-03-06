import os
import gc

import pandas as pd
from tqdm import tqdm

from load_config import config_WF

def process_all_datasets(config: dict):
    """
    Does the main data processing.
    """
    print("\nProcessing Wind Farm dataset.")
    for subtask in config['wind_farm']['subtask_list']:
        # augment conigurations with additional information
        config_wind = config_WF(config, subtask)
        # load data of this subtask
        df_data, df_locations = load_data(config_wind)
        # expand timestamp
        df_data[['hour', 'minute']] = df_data.Tmstamp.str.split(':', expand=True)
        df_data.drop(columns=['Tmstamp'], inplace=True)
        # expand data with sliding time window
        df_data = create_datapoints(config_wind, df_data)
        # split the loaded dataframe into training, validation and testing
        #split_train_val_test(config_wind, df_data, df_locations)
        
        
def load_data(config_wind: dict) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads the SCADA data and the location data of each turbine from csv files,
    and returns these.
    """    
    print('\nLoading data for {}!'.format(config_wind['subtask']))
    df_locations = pd.read_csv(config_wind['path_to_turb_loc_file'])
    if config_wind['subtask'] == 'days_245':
        # all data is in single file
        df_data = pd.read_csv(config_wind['path_to_data_raw_file'])
    elif config_wind['subtask'] == 'days_183':
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
        for in_filename, out_filename in zip(list_of_files_in, list_of_files_out):
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
    return df_data, df_locations

    
def create_datapoints(config_wind: dict, df_data: pd.DataFrame) -> pd.DataFrame:
    """
    """
    print(df_data.columns)
    # get a list of all turbine IDs available in data
    turbine_list = list(set(df_data['TurbID']))
    # set number of maximum days
    n_days = df_data['Days'].max()
    # create a zero values array in the maximum size it can fill given no sparsity
    values_array = np.zeros((len(turbine_list) * n_days * 24 * 6 
        - config_wind['historic_window'] - config_wind['prediction_window'],
        len(config_wind['fseries_name_list']) * config_wind['historic_window']
        + 4 + config_wind['prediction_window']))
    # set a datapoint counter
    data_counter = 0
    # iterate over all turbine IDs
    for turbine_id in turbine_list:
        # get corresponding entries
        df_turbine = df_data.loc[df_data['TurbID']==turbine_id].copy()
        # sort by time
        df_turbine.sort_values(by=['Day', 'hour', 'minute'], inplace=True,
            ignore_index=True)
        # iterate over entries of df_turbine
        for i in range(config_wind['historic_window'], 
            len(df_turbine)-config_wind['prediction_window']):
            # set spatial and temporal values
            values_array[data_counter, 0] = turbine_id
            values_array[data_counter, 1] = df_turbine['Days'][i]
            values_array[data_counter, 2] = df_turbine['hour'][i]
            values_array[data_counter, 3] = df_turbine['minute'][i]
            col_counter = 4
            # iterate over historic time window
            for j in range(i-config_wind['historic_window'], i):
                # iterate over time series feature names
                for colname in config_wind['fseries_name_list']:
                    values_array[data_counter, col_counter] = (
                        df_turbine[colname][j])
                    # increment column counter
                    col_counter += 1
            # iterate over prediction window
            for j in range(i, i+ config_wind['prediction_window']):
                values_array[data_counter, col_counter] = df_turbine['Patv'][j]
                # increment column counter
                col_counter += 1
            # increment counter
            data_counter += 1
    
    # create column name
    col_name_list = ['TrbID', 'day', 'hour', 'minute']
    new_fseries_name_list = ['wind_speed', 'wind_direction', 'temperature_out',
    'temperature_in', 'nacelle_angle', 'blade1_angle', 'blade2_angle',
    'blade3_angle', 'reactive_power', 'active_power']
    for i in range(config_wind['historic_window']+1):
        for colname_base in new_fseries_name_list:
            colname = colname_base + '_{}'.format(i)
            col_name_list.append(colname)
    
    # create dataframe and overwrite old one
    df_data = pd.DataFrame(values_array, columns=col_name_list)      
    # drop zero entries
    df_data.loc[~(df_data==0).all(axis=1)]
    return df_data
    
    
def split_train_val_test(config_wind: dict, df_data: pd.DataFrame,
    df_locations: pd.DataFrame):
    """
    """
    
    ###
    # Split training and ood testing
    ###
    
    # get spliting rules
    temporal_ood = config_wind['temporal_ood']
    # split of temporal ood
    df_test = df_data.loc[
        (df_data['day'].isin(temporal_ood['days_test']))
        | (df_data['hour'].isin(temporal_ood['hours_test']))
        | (df_data['minute'].isin(temporal_ood['minutes_test']))]
    # drop separated indices
    df_data = df_data.drop(df_test.index)
    # get spliting rules
    spatial_ood = config_wind['spatial_ood']
    # split of temporal ood
    df_spatial_test = df_data.loc[
        df_data['TurbID'].isin(spatial_ood['turbines_test'])]
    # drop separated indices
    df_data = df_data.drop(df_spatial_test.index)
    # concat to test
    df_test = pd.concat([df_test, df_spatial_test], ignore_index=True)
    # free up memory
    del df_spatial_test
    gc.collect()
    
    ###
    # Augment dataframes with location data
    ###
    
    # merge
    df_data = pd.merge(df_data, df_locations, on='TurbID', how='left')
    df_test = pd.merge(df_test, df_locations, on='TurbID', how='left')
    df_data.drop(columns=['TurbID'], inplace=True)
    df_test.drop(columns=['TurbID'], inplace=True)
    # free up memory
    del df_locations
    gc.collect()
    
    print(df_data.columns)
    print(df_test.columns)
    
    ###
    # Create data points
    ###    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
