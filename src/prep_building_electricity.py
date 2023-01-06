import pandas as pd
import os


def prepare_building_electricity_data(HYPER):

    """ """
    
    # import data first
    df_consumption, df_building_images, df_meteo_dict = import_all_data(HYPER)

    # process all data into desired format
    df_consumption_new, df_building_images_new = process_all_data(
        HYPER, 
        df_consumption, 
        df_building_images, 
        df_meteo_dict
    )
    


def process_all_data(
    HYPER, 
    df_consumption, 
    df_building_images, 
    df_meteo_dict
):
    
    """ """
    
    ###
    # Process consumption data ###
    ###
    
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
        
        
    # create a new dataframe you want to fill
    df_consumption_new = pd.DataFrame(columns=new_df_columns)
    
    # drop the year entries
    df_consumption.drop(index=1, inplace=True)
    
    # get corresponding time stamps series and reset indices
    time_stamps = df_consumption['building ID'].iloc[1:].reset_index(drop=True)
    
    # create a list of all building IDs
    building_id_list = list(df_consumption.columns.values[1:])
    
    # shorten for test
    building_id_list = building_id_list[:10]
    
    # declare df row counter
    counter_df_row = 0
    
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
        
            # declare empty list to save all entries of a single data point
            entry = []
            
            # Add temporal and spatial features to entry. Ensures same order as new_df_columns
            for entry_name in new_df_columns_base:
                command = 'entry.append({})'.format(entry_name)
                exec(command)
                
            # add meteorological data to entry
            for meteo_name, meteo_profile in meteo_dict.items():
                entry+= list(meteo_profile)
                
            # add load profile to entry
            entry+= list(load_profile)
            
            # append to dataframe
            df_consumption_new.loc[counter_df_row] = entry
    
            # increment df row counter
            counter_df_row += 1
            
    
    ###
    # Process building image data ###
    ###    
    
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
    
   
    ###
    # Save data
    ###
    
    # create saving path for building imagery
    saving_path = (
        HYPER.PATH_TO_DATA_BUILDING_ELECTRICITY_ADD 
        + 'building_images_pixel_histograms_rgb.csv'
    )
    
    # save df_building_images_new
    df_building_images_new.to_csv(saving_path, index=False)
    
    
    
    return df_consumption_new, df_building_images_new




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
    

