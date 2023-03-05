

import pandas as pd

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
        
        
def load_data(config_wind: dict) -> pandas.DataFrame:
    """
    """    
    
        
        
    # placeholder
    df_data = pd.DataFrame()
    return df_data
