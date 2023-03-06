import os

import pandas as pd
from tqdm import tqdm

from load_config import config_WF

def process_all_datasets(config: dict):
    """
    """
    print("\nProcessing Wind Farm dataset.")
    for subtask in config['wind_farm']['subtask_list']:
        # augment conigurations with additional information
        config_wind = config_WF(config, subtask)
        # load data of this subtask
        df_data, df_locations = load_data(config_wind)
        # Split the loaded dataframe into training, validation and testing
        split_train_val_test(config_wind, df_data, df_locations)
        
def load_data(config_wind: dict) -> (pd.DataFrame, pd.DataFrame):
    """
    """    
    print('\nLoading data for Wind Farm task!')
    df_locations = pd.read_csv(config_wind['path_to_turb_loc_file'])
    if config_wind['subtask'] == 'compete_train':
        # all data is in single file
        df_data = pd.read_csv(config_wind['path_to_data_raw_file'])
    elif config_wind['subtask'] == 'compete_test':
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
            #update progress bar
            pbar.update(1)
    return df_data, df_locations
    
def split_train_val_test(config_wind: dict, df_data: pd.DataFrame,
    df_locations: pd.DataFrame):
    """
    """
    pass
    
    
    
