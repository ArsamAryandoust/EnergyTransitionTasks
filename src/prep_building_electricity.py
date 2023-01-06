import pandas as pd
import os
import random

def process_all_data(HYPER):
    
    """ """
    
    raw_data_dict = import_consumption_profiles(HYPER)
    
    
    



def import_data(HYPER):
    
    """ """
    
    # import all electric consumption profiles
    df_consumption = pd.read_csv(HYPER.PATH_TO_BUILDING_YEAR_PROFILES_FILE)
    
    # import image pixel histogram values
    df_building_images = pd.read_csv(HYPER.PATH_TO_AERIAL_IMAGERY_FILE)
    
    # create path to sample meteo files
    meteo_filename_list = os.listdir(HYPER.PATH_TO_METEO_DATA_FOLDER)
    sample_filename = random.sample(meteo_filename_list, 1)[0]
    path_to_meteo_file = HYPER.PATH_TO_METEO_DATA_FOLDER + sample_filename
    
    # import sample meteo file
    df_sample_meteo_data = pd.read_csv(path_to_meteo_file)
    
    return df_consumption, df_building_images, df_sample_meteo_data
