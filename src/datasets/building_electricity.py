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
    
    # iterated over all subtasks
    for subtask in config['building_electricity']['subtask_list']:
        
        # augment config with currently iterated subtask paths
        config = config_BE(config, subtask)
        
        # import all data
        df_consumption, df_building_images, df_meteo_dict = import_all_data(
            config['building_electricity']
        )

        # process building imagery
        process_building_imagery(config['building_electricity'], df_building_images)
        
        # process meteo data and load profiles
        process_meteo_and_load_profiles(config, df_consumption, df_meteo_dict)
        
        # empty memory
        del df_consumption, df_building_images, df_meteo_dict
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
        config['path_to_data_building_electricity_add']
        + 'building_images_pixel_histograms_rgb.csv'
    )
    
    # save df_building_images_new
    df_building_images_new.to_csv(saving_path, index=False)

    
def process_meteo_and_load_profiles(
    config: dict, 
    df_consumption: pd.DataFrame,
    df_meteo_dict: pd.DataFrame
):
    """
    Main data processing.
    """
    
    # create new df column format
    new_df_columns_base = ['year', 'month', 'day', 'hour', 'quarter_hour', 'building_id']
    
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
        
        
    # drop the year entries
    df_consumption.drop(index=1, inplace=True)
    
    # get corresponding time stamps series and reset indices
    time_stamps = df_consumption['building ID'].iloc[1:].reset_index(drop=True)
    
    # create a list of all building IDs
    building_id_list = list(df_consumption.columns.values[1:])
    
    # declare df row counter
    counter_df_row = 0
    
    # decleare empty values array
    values_array = np.zeros(
        (
            len(building_id_list) * math.floor(
                365 - config['building_electricity']['historic_window'] / 96 
            ),
            (
                len(new_df_columns_base) 
                + config['building_electricity']['historic_window'] * len(config['building_electricity']['meteo_name_list'])
                + config['building_electricity']['prediction_window']
            )
        )
    )
    
    # create progress bar
    pbar = tqdm(total=len(building_id_list))
    
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
        for i in range(config['building_electricity']['historic_window'], len(time_stamps), config['building_electricity']['prediction_window']):
            
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
                meteo_values = df_meteo[meteo_name][(i-config['building_electricity']['historic_window']):i].values
                meteo_dict[meteo_name] = meteo_values
            
            # get iterated load profile data
            load_profile = building_load[i:(i+config['building_electricity']['prediction_window'])].values
            
            # Add features to values_array. Ensures same order as new_df_columns.
            for index_df_col, entry_name in enumerate(new_df_columns_base):
                command = 'values_array[counter_df_row, index_df_col] = {}'.format(entry_name)
                exec(command)
                
            # add meteorological data to entry
            for meteo_name, meteo_profile in meteo_dict.items():
                for i in range(len(meteo_profile)):
                    index_df_col += 1
                    values_array[counter_df_row, index_df_col] = meteo_profile[i]
                
            # add load profile to entry
            for i in range(len(load_profile)):
                index_df_col += 1
                values_array[counter_df_row, index_df_col] = load_profile[i]
            
    
            # increment df row counter
            counter_df_row += 1
    
        # increment progbar
        pbar.update(1) 
            
    # create a new dataframe you want to fill
    df_consumption_new = pd.DataFrame(data=values_array, columns=new_df_columns)
    
    # get total number of data points 
    n_data_total = len(df_consumption_new)
    
    # test split
    df_testing = df_consumption_new.sample(frac=config['building_electricity']['test_split'], random_state=config['general']['seed'])
    
    # drop indices taken for testing from remaining data
    df_consumption_new = df_consumption_new.drop(df_testing.index)
    
    # do training split
    df_training = df_consumption_new.sample(frac=config['building_electricity']['train_val_split'], random_state=config['general']['seed'])
    df_validation = df_consumption_new.drop(df_training.index)
    
    print(
        "Training data   :    {:.0%} \n".format(len(df_training)/n_data_total),
        "Validation data :    {:.0%} \n".format(len(df_validation)/n_data_total),
        "Testing data    :    {:.0%} \n".format(len(df_testing)/n_data_total)
    )
    
    # save results
    saving_path = config['building_electricity']['path_to_data_building_electricity_train'] + 'training_data.csv'
    df_training.to_csv(saving_path, index=False)
    
    saving_path = config['building_electricity']['path_to_data_building_electricity_val'] + 'validation_data.csv'
    df_validation.to_csv(saving_path, index=False)
    
    saving_path = config['building_electricity']['path_to_data_building_electricity_test'] + 'testing_data.csv'
    df_testing.to_csv(saving_path, index=False)
    
    
