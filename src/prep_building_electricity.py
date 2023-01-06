import os
import pandas as pd
import numpy as np



def process_meteo_and_load_profiles(
    HYPER, 
    df_consumption,
    df_meteo_dict
):
    
    """ """
    
    # create new df column format
    new_df_columns_base = ['year', 'month', 'day', 'hour', 'quarter_hour', 'building_id']
    
    # fill a separate list 
    new_df_columns = new_df_columns_base.copy()
    
    # append column entries for meteorological data
    for column_name in HYPER.METEO_NAME_LIST:
        for pred_time_step in range(HYPER.PREDICTION_WINDOW):
            entry_name = '{}_{}'.format(column_name, pred_time_step+1)
            new_df_columns.append(entry_name)
    
    # append column entries for electric load
    for pred_time_step in range(HYPER.PREDICTION_WINDOW):
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
            len(building_id_list) * 365, 
            (
                len(new_df_columns_base) 
                + HYPER.PREDICTION_WINDOW * (len(HYPER.METEO_NAME_LIST) + 1)
            )
        )
    )
    
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
        for i in range(0, len(time_stamps), HYPER.PREDICTION_WINDOW):
            
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
            for meteo_name in HYPER.METEO_NAME_LIST:
                meteo_dict[meteo_name] = df_meteo[meteo_name][i:(i+HYPER.PREDICTION_WINDOW)].values
            
            # get iterated load profile data
            load_profile = building_load[i:(i+HYPER.PREDICTION_WINDOW)].values
        
            
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
    
            
    # create a new dataframe you want to fill
    df_consumption_new = pd.DataFrame(data=values_array, columns=new_df_columns)
    
    # test split
    df_testing = df_consumption_new.sample(frac=HYPER.TEST_SPLIT, random_state=HYPER.SEED)
    
    # drop indices taken for testing from remaining data
    df_consumption_new = df_consumption_new.drop(df_testing.index)
    
    # do training split
    df_training = df_consumption_new.sample(frac=HYPER.TRAIN_VAL_SPLIT, random_state=HYPER.SEED)
    df_validation = df_consumption_new.drop(df_training.index)
    
    
    
    return df_training, df_validation, df_testing


def process_building_imagery(HYPER, df_building_images):

    """ """
    
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
        HYPER.PATH_TO_DATA_BUILDING_ELECTRICITY_ADD 
        + 'building_images_pixel_histograms_rgb.csv'
    )
    
    # save df_building_images_new
    df_building_images_new.to_csv(saving_path, index=False)

    return df_building_images_new


def import_all_data(HYPER):
    
    """ """
    
    # import all electric consumption profiles
    df_consumption = pd.read_csv(HYPER.PATH_TO_RAW_BUILDING_YEAR_PROFILES_FILE)
    
    # import image pixel histogram values
    df_building_images = pd.read_csv(HYPER.PATH_TO_RAW_AERIAL_IMAGERY_FILE)
    
    # create path to sample meteo files
    meteo_filename_list = os.listdir(HYPER.PATH_TO_RAW_METEO_DATA_FOLDER)
    
    # decleare empty dictionary for saving all meteo dataframes
    df_meteo_dict = {}
    
    # iterate over all filenames
    for filename in meteo_filename_list:
        
        # create full path to iterated file
        path_to_meteo_file = HYPER.PATH_TO_RAW_METEO_DATA_FOLDER + filename
        
        # import meteorological data
        df_meteo = pd.read_csv(path_to_meteo_file)
        
        # save imported dataframe to dataframe dictionary
        df_meteo_dict[filename] = df_meteo
        
    
    return df_consumption, df_building_images, df_meteo_dict
    

