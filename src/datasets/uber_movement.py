import math
import gc
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from load_config import config_UM

def process_all_datasets(config: dict):
    """
    Processes all datasets for Uber Movement prediction task.
    """
    
    print("Processing Uber Movement dataset.")
    
    # iterated over all subtasks
    for subtask in config['uber_movement']['subtask_list']:
        # augment conigurations with additional information
        config = config_UM(config, subtask)
    
        # process geographic data
        save_city_id_mapping(config['uber_movement'])
        process_geographic_information(config['uber_movement'])
    
        # split training validation testing
        split_train_val_test(config)
    
    
def save_city_id_mapping(config: str):
    """
    Creates a dataframe from dictionary of city to ID mapping and saves it.
    """
    
    # create dataframe from dictionary
    df = pd.DataFrame.from_dict(
        config['city_id_mapping'], 
        orient='index', 
        columns=['city_id']
    )
    
    # save file
    saving_path = config['path_to_data_add'] + 'city_to_id_mapping.csv'
    df.to_csv(saving_path)
    
    
def process_geographic_information(config: dict):
    """
    Processes and saves geographic features of cities and their zones.
    """
    
    # iterate over all cities
    for city in config['list_of_cities']:
    
        # import geojson for iterated city
        df_geojson = import_geojson(config, city)
        
        # extract geojson information of city zones as latitude and longitude df
        df_latitudes, df_longitudes = process_geojson(df_geojson)
        
        ### Transform lat and long coordinates into unit sphere coordinate system
        
        # calculate values you need for 
        df_sin_lon = df_longitudes.applymap(sin_transform)
        df_sin_lat = df_latitudes.applymap(sin_transform)
        df_cos_lat = df_latitudes.applymap(cos_transform)
        df_cos_lon = df_longitudes.applymap(cos_transform)

        # calculate coordaintes
        df_x_cord = df_cos_lat.mul(df_cos_lon)
        df_y_cord = df_cos_lat.mul(df_sin_lon)
        df_z_cord = df_sin_lat
        
        
        ### Transform column names ###
        
        # transform x_cord columns
        col_list = df_x_cord.columns.to_list()
        new_col_list = transform_col_names(col_list, 'x_cord')
        df_x_cord.columns = new_col_list
        
        # transform y_cord columns
        col_list = df_y_cord.columns.to_list()
        new_col_list = transform_col_names(col_list, 'y_cord')
        df_y_cord.columns = new_col_list
        
        # transform z_cord columns
        col_list = df_z_cord.columns.to_list()
        new_col_list = transform_col_names(col_list, 'z_cord')
        df_z_cord.columns = new_col_list
        
        
        ### Save into one csv file ###
        
        # concatenate all dataframes
        df_geographic_info = pd.concat([df_x_cord, df_y_cord, df_z_cord], axis=1)
        filename = city + '.csv'
        saving_path = (config['path_to_data_add'] + filename)
        df_geographic_info.to_csv(saving_path)
        

def import_geojson(config: dict, city: str) -> pd.DataFrame:
    """ 
    Uses the city to file mapping of city to load the geo-json file and returns
    it as a dataframe.
    """
    files_dict = config['city_files_mapping'][city]
    path_to_json = config['path_to_data_raw'] + city + '/' + files_dict['json']
    df_geojson = pd.read_json(path_to_json)
    return df_geojson
    
    
def process_geojson(df_geojson: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """ Maps Uber Movement city zone IDs to a flattened list of latitude and 
    longitude coordinates in the format of two dictionaries. Uses the recursive 
    function called foster_coordinates_recursive to flatten the differently nested 
    data.
    """
    
    df_geojson.pop('type')
    df_geojson = df_geojson['features']
    
    map_json_entry_to_movement_id = dict()

    for json_id, json_entry in enumerate(df_geojson):
        
        map_json_entry_to_movement_id[json_id] = int(
          json_entry['properties']['MOVEMENT_ID']
        )
    
    map_movement_id_to_latitude_coordinates = dict()
    map_movement_id_to_longitude_coordinates = dict()

    for k, v in map_json_entry_to_movement_id.items():
        map_movement_id_to_latitude_coordinates[v] = []
        map_movement_id_to_longitude_coordinates[v] = []


    for json_id, movement_id in map_json_entry_to_movement_id.items():
        coordinates = df_geojson[json_id]['geometry']['coordinates']
        
        (
            map_movement_id_to_latitude_coordinates, 
            map_movement_id_to_longitude_coordinates
        ) = foster_coordinates_recursive(
            movement_id,
            map_movement_id_to_latitude_coordinates,
            map_movement_id_to_longitude_coordinates,
            coordinates
        )
        
    df_latitudes = pd.DataFrame.from_dict(
        map_movement_id_to_latitude_coordinates, 
        orient='index'
    ).transpose()
    
    df_longitudes = pd.DataFrame.from_dict(
        map_movement_id_to_longitude_coordinates, 
        orient='index'
    ).transpose()
    
    return df_latitudes, df_longitudes
  
  
def foster_coordinates_recursive(
    movement_id: int,
    map_movement_id_to_latitude_coordinates: dict,
    map_movement_id_to_longitude_coordinates: dict,
    coordinates: pd.Series
) -> tuple:

    """ Flattens the coordinates of a passed city zone id (movement_id)
    and coordiates list recursively and saves their numeric values
    in the dictionaries that map movement ids to a list of latitude and 
    longitude coordinates.
    """

    dummy = 0

    for j in coordinates:

        if type(j) != list and dummy == 0:

            map_movement_id_to_longitude_coordinates[movement_id].append(j)
            dummy = 1
            continue

        elif type(j) != list and dummy == 1:

            map_movement_id_to_latitude_coordinates[movement_id].append(j)
            break

        else:

            dummy = 0
            coordinates = j
            (
                map_movement_id_to_latitude_coordinates,
                map_movement_id_to_longitude_coordinates
            ) = foster_coordinates_recursive(
                movement_id,
                map_movement_id_to_latitude_coordinates,
                map_movement_id_to_longitude_coordinates,
                coordinates
            )

    map_movement_id_to_coordinates = (
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    )

    return map_movement_id_to_coordinates
    
    
def degree_to_phi(degree_latlon: float):
    """ 
    transform degrees into radiants 
    """
    return degree_latlon / 180 * math.pi


def cos_transform(degree_latlon: float):
    """ 
    Transform degrees into radiants and return cosine value. 
    """
    phi_latlon = degree_to_phi(degree_latlon)
    return np.cos(phi_latlon)


def sin_transform(degree_latlon: float):
    """
    Transform degrees into radiants and return sine value. 
    """
    phi_latlon = degree_to_phi(degree_latlon)
    return np.sin(phi_latlon)
    
    
def transform_col_names(col_list: list, name_base: str) -> list:
    """ 
    Transform column name list by adding the past name base as prefix.
    """
    new_col_list = [name_base + '_' + str(entry) for entry in col_list]
    return new_col_list
    

def split_train_val_test(config: dict):
    """ 
    Splits and saves datasets according to configuration rules.
    """
    
    config_uber = config['uber_movement']
    # create progress bar
    pbar = tqdm(total=len(config_uber['list_of_cities']))
    
    # decleare empty dataframes for trainining validation and testing
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    
    # declare data point counters
    train_chunk_counter, val_chunk_counter, test_chunk_counter = 0, 0, 0
    
    # iterate over all available cities
    for city in config_uber['list_of_cities']:
        
        # tell us whats going on
        print('Processing data for', city)
        
        # check if city is in testing city list
        if city in (
            config_uber['test_split_dict']['spatial_dict']['list_of_cities_test']
        ):
            testing_city = True
        else:
            testing_city = False
        
        # import all csv files for currently iterated city
        df_csv_dict_list = import_csvdata(config_uber, city)
        
        # iterate over all imported csv files for this city
        first_iteration = True
        for df_csv_dict in df_csv_dict_list:
            # check if testing year
            if df_csv_dict['year'] == (
                config_uber['test_split_dict']['temporal_dict']['year']
            ):
                testing_year = True
            else:
                testing_year = False
                
            # check if testing quarter of year
            if df_csv_dict['quarter_of_year'] == (
                config_uber['test_split_dict']['temporal_dict']['quarter_of_year']
            ):
                testing_quarter = True
            else:
                testing_quarter = False
            
            # augment csv
            df_augmented_csvdata = process_csvdata(config, df_csv_dict, city)
            
            # free up memory     
            del df_csv_dict['df']
            gc.collect()
            
            # get the subset of city zones for test splits once per city
            if first_iteration:
                n_city_zones = max(
                    df_augmented_csvdata['source_id'].max(),
                    df_augmented_csvdata['destination_id'].max()
                )
                
                # get number of test city zones you want to split
                n_test_city_zones = round(
                    n_city_zones * config_uber['spatial_test_split']
                )
                
                # randomly sample test city zones
                random.seed(config['general']['seed'])
                test_city_zone_list = random.sample(
                    range(n_city_zones), 
                    n_test_city_zones
                )
                
                # set false so as to not enter branch anymore
                first_iteration= False
            
            if testing_city or testing_year or testing_quarter:
                # append all data to test dataframe
                df_test = pd.concat([df_test, df_augmented_csvdata])
                
                # free up memory     
                del df_augmented_csvdata   
                gc.collect()
                
            else:
                # extract rows from dataframe with matching city zones
                df_test_city_zones = df_augmented_csvdata.loc[
                    (df_augmented_csvdata['destination_id'].isin(test_city_zone_list)) 
                    | (df_augmented_csvdata['source_id'].isin(test_city_zone_list))
                ]
                
                # set the remaining rows for training and validation
                df_augmented_csvdata = df_augmented_csvdata.drop(
                    df_test_city_zones.index
                )
                
                # append to test dataframe
                df_test = pd.concat([df_test, df_test_city_zones])
                
                # free up memory
                del df_test_city_zones
                gc.collect()
                
                # extract the rows from dataframe with matching hours of day for test
                df_test_hours_of_day = df_augmented_csvdata.loc[
                    df_augmented_csvdata['hour_of_day'].isin(
                        config_uber['test_split_dict']['temporal_dict']['hours_of_day']
                    )
                ]
                
                # set the remaining rows for training and validation
                df_augmented_csvdata = df_augmented_csvdata.drop(
                    df_test_hours_of_day.index
                )
                
                # append to test dataframe
                df_test = pd.concat([df_test, df_test_hours_of_day])
                
                # free up memory
                del df_test_hours_of_day
                gc.collect()
                
                # split off validation data
                df_val_append = df_test.sample(
                    frac=config_uber['val_test_split'], 
                    random_state=config['general']['seed']
                )
                
                # remove validation data from test
                df_test = df_test.drop(df_val_append.index)
                
                # append to validation dataframe
                df_val = pd.concat([df_val, df_val_append])
                
                # free up memory     
                del df_val_append   
                gc.collect()
                
                # append remaining data to training dataset
                df_train = pd.concat([df_train, df_augmented_csvdata])
                
                # free up memory     
                del df_augmented_csvdata   
                gc.collect()
                
            
            ### Save resulting data in chunks
            df_train, train_chunk_counter = save_chunk(
                config,
                df_train,
                train_chunk_counter,
                config_uber['path_to_data_train'],
                'training_data'    
            )
            df_val, val_chunk_counter = save_chunk(
                config,
                df_val,
                val_chunk_counter,
                config_uber['path_to_data_val'],
                'validation_data'
            )
            df_test, test_chunk_counter = save_chunk(
                config,
                df_test,
                test_chunk_counter,
                config_uber['path_to_data_test'],
                'testing_data'
            )
            
        # update progress bar
        pbar.update(1)

    ### Tell us the ratios that result from our splitting rules
    n_train = (
        train_chunk_counter * config_uber['datapoints_per_file'] 
        + len(df_train.index)
    )
    n_val = (
        val_chunk_counter * config_uber['datapoints_per_file'] 
        + len(df_val.index)
    )
    n_test = (
        test_chunk_counter * config_uber['datapoints_per_file'] 
        + len(df_test.index)
    )
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
    
    ### Save results of last iteration
    df_train, train_chunk_counter = save_chunk(
        config,
        df_train,
        train_chunk_counter,
        config_uber['path_to_data_train'],
        'training_data',
        last_iteration=True  
    )
    df_val, val_chunk_counter = save_chunk(
        config,
        df_val,
        val_chunk_counter,
        config_uber['path_to_data_val'],
        'validation_data',
        last_iteration=True  
    )
    df_test, test_chunk_counter = save_chunk(
        config,
        df_test,
        test_chunk_counter,
        config_uber['path_to_data_test'],
        'testing_data',
        last_iteration=True  
    )
    
    
def import_csvdata(config: dict, city: str):
    """ 
    Imports the Uber Movement data for a passed city 
    """
    
    files_dict = config['city_files_mapping'][city]
    df_csv_dict_list = []
    for csv_file_dict in files_dict['csv_file_dict_list']:
        path_to_csv = (
            config['path_to_data_raw'] + city + '/' + csv_file_dict['filename']
        )
        df_csv = pd.read_csv(path_to_csv)
        csv_df_dict = csv_file_dict.copy()
        csv_df_dict['df'] = df_csv
        df_csv_dict_list.append(csv_df_dict)
    
    return df_csv_dict_list
    
    
def process_csvdata(config: dict, df_csv_dict: pd.DataFrame, city: str):
    """ 
    """
    
    # copy raw dataframe
    df_augmented = df_csv_dict['df']
    
    # subsample or shuffle data (for frac=1)    
    df_augmented = df_augmented.sample(
        frac=config['uber_movement']['subsample_frac'],
        random_state=config['general']['seed']
    )
    
    # augment raw dataframe
    df_augmented.insert(
        0, 
        'city_id', 
        config['uber_movement']['city_id_mapping'][city]
    )
    df_augmented.insert(
        3, 
        'year', 
        df_csv_dict['year']
    )
    df_augmented.insert(
        4, 
        'quarter_of_year', 
        df_csv_dict['quarter_of_year']
    )
    df_augmented.insert(
        5, 
        'daytype', 
        df_csv_dict['daytype']
    )
    
    # rename some columns with more clear names
    df_augmented.rename(
        columns={
            'hod':'hour_of_day', 
            'sourceid':'source_id', 
            'dstid':'destination_id'
        }, 
        inplace=True
    )
    
    # remove any rows with nan entry
    df_augmented = df_augmented[df_augmented.isnull().sum(axis=1) < 1]
    
    return df_augmented
    
def save_chunk(
    config: dict,
    df: pd.DataFrame,
    chunk_counter: int,
    saving_path: str,
    filename: str,
    last_iteration=False 
) -> (pd.DataFrame, int):
    """ 
    Save a chunk of data and return remaining with chunk counter 
    """
    
    while (
        len(df.index) > config['uber_movement']['datapoints_per_file'] 
        or last_iteration
    ):
        # increment chunk counter 
        chunk_counter += 1
        
        # create path to saving
        path_to_saving = (
            saving_path
            + filename
            + '_{}.csv'.format(chunk_counter)
        )
        
        # shuffle dataframe
        df = df.sample(frac=1, random_state=config['general']['seed'])
        
        # save chunk
        df.iloc[:config['uber_movement']['datapoints_per_file']].to_csv(
            path_to_saving, 
            index=False
        )
        
        # delete saved chunk
        df = df[config['uber_movement']['datapoints_per_file']:]
            
        # Must be set to exit loop on last iteration
        last_iteration = False
        
    return df, chunk_counter
    
