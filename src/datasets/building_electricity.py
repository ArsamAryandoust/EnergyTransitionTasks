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
    for subtask in config['building_electricity']['subtask_list']:
        # augment config with currently iterated subtask paths
        config = config_BE(config, subtask)
        
        # import all data
        (
            df_consumption, 
            df_building_images, 
            df_meteo_dict
        ) = import_all_data(config['building_electricity'])

        # process building imagery
        process_building_imagery(
            config['building_electricity'], 
            df_building_images
        )

        # free up memory
        del df_building_images
        gc.collect()
                
        # process meteo data and load profiles
        df_dataset = process_meteo_and_load_profiles(
            config, 
            df_consumption, 
            df_meteo_dict
        )
        
        # free up memory
        del df_consumption, df_meteo_dict
        gc.collect()
        
        # Do trainining, validation and testing split
        split_train_val_test(config, df_dataset)
        
        # free up memory
        del df_dataset
        gc.collect()
    
    
def import_all_data(config: dict) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Imports electric consumption profiles profiles, building imagery pixel data,
    and meteorological data.
    """
    
    # import all electric consumption profiles
    df_consumption = pd.read_csv(config['path_to_raw_building_year_profiles_file'])
    
    # import image pixel histogram values
    df_building_images = pd.read_csv(config['path_to_raw_aerial_imagery_file'])
    
    # create path to sample meteo files
    meteo_filename_list = os.listdir(config['path_to_raw_meteo_data_folder'])
    
    # decleare empty dictionary for saving all meteo dataframes
    df_meteo_dict = {}
    
    # iterate over all filenames
    for filename in meteo_filename_list:
        # create full path to iterated file
        path_to_meteo_file = config['path_to_raw_meteo_data_folder'] + filename
        
        # import meteorological data
        df_meteo = pd.read_csv(path_to_meteo_file)
        
        # save imported dataframe to dataframe dictionary
        df_meteo_dict[filename] = df_meteo
        
    return df_consumption, df_building_images, df_meteo_dict
    

def process_building_imagery(config: dict, df_building_images: pd.DataFrame):
    """
    Simply changes the column name of aerial imagery histograms of buildings
    by adding the pre-fix 'building_' to IDs and saves file with new column names.
    """
    
    # get list of columns
    columns_df_list = df_building_images.columns
    
    # declare empty list to fill
    new_columns_list = []
    
    # iterate over all column names
    for entry in columns_df_list:
        # create new entry
        new_entry = 'building_{}'.format(entry)
        
        # append new entry to new column list
        new_columns_list.append(new_entry)
    
    # copy old dataframe 1 to 1    
    df_building_images_new = df_building_images
    
    # only replace its column names
    df_building_images_new.columns = new_columns_list
    
    # create saving path for building imagery
    saving_path = (
        config['path_to_data_add']
        + 'building_images_pixel_histograms_rgb.csv'
    )
    
    # save df_building_images_new
    df_building_images_new.to_csv(saving_path, index=False)

    
def process_meteo_and_load_profiles(
    config: dict, 
    df_consumption: pd.DataFrame,
    df_meteo_dict: pd.DataFrame
) -> pd.DataFrame:
    """
    Main data processing. Takes electric load profiles and meteorological data
    and combines these into a single dataset dataframe.
    """
    
    ###
    # Create dataframe column
    ###
    
    # create new df column format
    new_df_columns_base = [
        'year', 'month', 'day', 'hour', 
        'quarter_hour', 'building_id'
    ]
    
    # fill a separate list 
    new_df_columns = new_df_columns_base.copy()
    
    # append column entries for meteorological data
    for column_name in config['building_electricity']['meteo_name_list']:
        for pred_time_step in range(config['building_electricity']['historic_window']):
            entry_name = '{}_{}'.format(column_name, pred_time_step+1)
            new_df_columns.append(entry_name)
    
    # append column entries for electric load
    for pred_time_step in range(config['building_electricity']['prediction_window']):
        entry_name = 'load_{}'.format(pred_time_step+1)
        new_df_columns.append(entry_name)
        
    ###
    # Create datasets
    ###
        
    # drop the year entries
    df_consumption.drop(index=1, inplace=True)
    
    # get corresponding time stamps series and reset indices
    time_stamps = df_consumption['building ID'].iloc[1:].reset_index(drop=True)
    
    # create a list of all building IDs
    building_id_list = list(df_consumption.columns.values[1:])
    
    
    # decleare empty values array. Filling matrix pre-allocates memory and decreases
    # computational time significantly.
    values_array = np.zeros(
        (
            len(building_id_list) * (
                len(time_stamps)
                - config['building_electricity']['historic_window']
                - config['building_electricity']['prediction_window']
            ),
            (
                len(new_df_columns_base) 
                + config['building_electricity']['historic_window'] * (
                    len(config['building_electricity']['meteo_name_list'])
                )
                + config['building_electricity']['prediction_window']
            )
        )
    )
    
    # create progress bar
    pbar = tqdm(total=len(building_id_list))
    
    # declare df row counter
    datapoint_counter = 0
    
    # iterate over all building IDs
    for building_id in building_id_list:
        # get cluster id as integer
        cluster_id = df_consumption[building_id].iloc[0].astype(int)
        
        # get building load with new indices
        building_load = df_consumption[building_id].iloc[1:].reset_index(drop=True)
        
        # transform building id into integer
        building_id = int(building_id)

        # create key to corresponding meteo data
        key_meteo = 'meteo_{}_2014.csv'.format(cluster_id)
        
        # get corresponding meteorological data
        df_meteo = df_meteo_dict[key_meteo]
        
        # drop local_time column 
        df_meteo = df_meteo.drop(columns=['local_time'])
        
        # iterate over all time stamps in prediction window steps
        for i in range(
            config['building_electricity']['historic_window'], 
            len(time_stamps) - config['building_electricity']['prediction_window']
        ):
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
            for meteo_name in config['building_electricity']['meteo_name_list']:
                meteo_values = df_meteo[meteo_name][
                    (i-config['building_electricity']['historic_window']):i
                ].values
                meteo_dict[meteo_name] = meteo_values
            
            # get iterated load profile data
            load_profile = building_load[
                i:(i+config['building_electricity']['prediction_window'])
            ].values
            
            # add features to values_array. Ensures same order as new_df_columns.
            for index_col, entry_name in enumerate(new_df_columns_base):
                command = 'values_array[datapoint_counter, index_col] = {}'.format(
                    entry_name
                )
                exec(command)
                
            # add meteorological data to entry
            for meteo_name, meteo_profile in meteo_dict.items():
                for i in range(len(meteo_profile)):
                    index_col += 1
                    values_array[datapoint_counter, index_col] = meteo_profile[i]
                
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
    
    # free up memory
    del values_array
    gc.collect()
    
    return df_dataset
    
    
def split_train_val_test(config: dict, df_dataset: pd.DataFrame):
    """
    Splits and saves datasets according to configuration rules.
    """
    
    # get total number of data points 
    n_data_total = len(df_dataset)
    
    # get the out-of-distribution splitting rules
    temporal_ood = config['building_electricity']['ood_split_dict']['temporal_dict']
    spatial_ood = config['building_electricity']['ood_split_dict']['spatial_dict']
    
    # create spatial ood split
    df_testing = df_dataset.loc[
        (df_dataset['building_id'].isin(spatial_ood['building_id_list']))
    ]
    
    # remove split spatial ood data points
    df_dataset = df_dataset.drop(df_testing.index)
    
    # create temporal ood split
    df_temporal_ood = df_dataset.loc[
        (df_dataset['month'].isin(temporal_ood['month_list']))
        | (df_dataset['day'].isin(temporal_ood['day_list']))
        | (df_dataset['hour'].isin(temporal_ood['hour_list']))
        | (df_dataset['quarter_hour'].isin(temporal_ood['quarter_hour_list']))
    ]
    
    # remove the spatial ood split data points which is training dataset
    df_training = df_dataset.drop(df_temporal_ood.index)
    
    # free up memory
    del df_dataset
    gc.collect()
    
    # append to testing dataset
    df_testing = pd.concat([df_testing, df_temporal_ood])
    
    # free up memory
    del df_temporal_ood
    gc.collect()
    
    # do validation split
    df_validation = df_testing.sample(
        frac=config['building_electricity']['val_test_split'], 
        random_state=config['general']['seed']
    )
    
    # remove validation data split from testing dataset
    df_testing = df_testing.drop(df_validation.index)
     
    # calculate and analyze dataset properties
    n_train, n_val, n_test = len(df_training), len(df_validation), len(df_testing)
    n_total = n_train + n_val + n_test
    
    print(
        "Training data   :   {}/{} {:.0%}".format(
              n_train, 
              n_total, 
              n_train/n_total
        ),
        "\nValidation data :   {}/{} {:.0%}".format(
            n_val,
            n_total,
            n_val/n_total
        ),
        "\nTesting data    :   {}/{} {:.0%}".format(
            n_test,
            n_total,
            n_test/n_total
        )
    )
    
    # save results in chunks
    save_in_chunks(
        config,
        config['building_electricity']['path_to_data_train'] + 'training_data', 
        df_training
    )
    save_in_chunks(
        config,
        config['building_electricity']['path_to_data_val'] + 'validation_data', 
        df_validation
    )
    save_in_chunks(
        config,
        config['building_electricity']['path_to_data_test'] + 'testing_data', 
        df_testing
    )

    
def save_in_chunks(config: dict, saving_path: str, df: pd.DataFrame):
    """
    Shuffles dataframe, then saves it in chunks with number of datapoints per 
    file defined by config such that each file takes less than about 1 GB size.
    """
    
    df = df.sample(frac=1, random_state=config['general']['seed'])
    for file_counter in range(1, 312321321312):
        path_to_saving = saving_path + '_{}.csv'.format(file_counter)
        df.iloc[
            :config['building_electricity']['datapoints_per_file']
        ].to_csv(path_to_saving, index=False)
        df = df[config['building_electricity']['datapoints_per_file']:]
        if len(df) == 0:
            break
            
            
