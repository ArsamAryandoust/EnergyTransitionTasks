import os

import pandas as pd
from tqdm import tqdm

from load_config import config_WF

def process_all_datasets(config: dict):
    """
    """
    print("Processing Wind Farm dataset.")
    for subtask in config['wind_farm']['subtask_list']:
        # augment conigurations with additional information
        config_wind = config_WF(config, subtask)
        # load data of this subtask
        df_data = load_data(config_wind)
        
        
def load_data(config_wind: dict) -> pd.DataFrame:
    """
    """    
    print('Loading data for Wind Farm task!')
    if config_wind['subtask'] == 'compete_train':
        # all data is in single file
        df_data = pd.read_csv(config_wind['path_to_data_raw_file'])
        print('\nLargest turbine ID is: {} \n'.format(df_data['TurbID'].max()))
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
            
        
    # placeholder
    df_data = pd.DataFrame()
    return df_data
