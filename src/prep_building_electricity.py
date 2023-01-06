


import pandas as pd


def process_all_data(HYPER):
    
    """ """
    
    raw_data_dict = import_consumption_profiles(HYPER)
    
    
    



def import_consumption_profiles(HYPER):
    
    """ """
    
    
    # load building year profiles
    df = pd.read_csv(HYPER.PATH_TO_BUILDING_YEAR_PROFILES_FILE)
    
    # get the building IDs of profiles
    building_ids = df.columns.values[1:]
    
    # get the cluster IDs of profiles and drop the row
    cluster_ids = df.iloc[0, 1:].values.astype(int)
    
    
    return raw_data_dict
