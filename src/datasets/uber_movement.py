import pandas as pd
import numpy as np

from load_config import config_UM

def process_all_datasets(config: dict):
    """
    Processes all datasets for Uber Movement prediction task.
    """
    # iterated over all subtasks
    for subtask in config['building_electricity']['subtask_list']:
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
        
        # create file name
        filename = city + '.csv'
        
        # create saving path
        saving_path = (
            HYPER.PATH_TO_DATA_UBERMOVEMENT_ADDITIONAL 
            + filename
        )
        
        # save dataframe
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
    
    
def degree_to_phi(degree_latlon):
    """ 
    transform degrees into radiants 
    """
    return degree_latlon / 180 * math.pi


def cos_transform(degree_latlon):
    """ 
    Transform degrees into radiants and return cosine value. 
    """
    phi_latlon = degree_to_phi(degree_latlon)
    return np.cos(phi_latlon)


def sin_transform(degree_latlon):
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
    
    # create progress bar
    pbar = tqdm(total=len(config['uber_movement']['list_of_cities']))
    
    # decleare empty dataframes for trainining validation and testing
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    
    # declare data point counters
    train_chunk_counter, val_chunk_counter, test_chunk_counter = 0, 0, 0
    
    # iterate over all available cities
    for city in HYPER.UBERMOVEMENT_LIST_OF_CITIES:
        
        # tell us whats going on
        print('Processing data for', city)
        
        # check if city is in testing city list
        if city in list_of_cities_test:
            testing_city = True
        else:
            testing_city = False
        
        # import all csv files for currently iterated city
        df_csv_dict_list = import_csvdata(HYPER, city)
        
        
        # iterate over all imported csv files for this city
        for iter_csv, df_csv_dict in enumerate(df_csv_dict_list):
        
        
            # check if testing year
            if df_csv_dict['year'] == HYPER.TEST_SPLIT_DICT_UBERMOVEMENT['temporal_dict']['year']:
                testing_year = True
            else:
                testing_year = False
                
            # check if testing quarter of year
            if df_csv_dict['quarter_of_year'] == HYPER.TEST_SPLIT_DICT_UBERMOVEMENT['temporal_dict']['quarter_of_year']:
                testing_quarter = True
            else:
                testing_quarter = False
            
            # augment csv
            df_augmented_csvdata = process_csvdata(HYPER, df_csv_dict, city)
            
            # free up memory     
            del df_csv_dict['df']
            gc.collect()
            
            # get the subset of city zones for test splits once per city
            if iter_csv == 0:
                n_city_zones = max(
                    df_augmented_csvdata['source_id'].max(),
                    df_augmented_csvdata['destination_id'].max()
                )
                
                # get number of test city zones you want to split
                n_test_city_zones = round(
                    n_city_zones * HYPER.TEST_SPLIT_DICT_UBERMOVEMENT['spatial_dict']['city_zone_share']
                )
                
                # randomly sample test city zones
                random.seed(HYPER.SEED)
                test_city_zone_list = random.sample(range(n_city_zones), n_test_city_zones)
            
            if testing_city or testing_year or testing_quarter:
                
                # append all data to test dataframe
                df_test = pd.concat([df_test, df_augmented_csvdata])
                
                # free up memory     
                del df_augmented_csvdata   
                gc.collect()
                
            else:
                
                # extract the rows from dataframe with matching city zones in origin and destination
                df_test_city_zones = df_augmented_csvdata.loc[
                    (df_augmented_csvdata['destination_id'].isin(test_city_zone_list)) 
                    | (df_augmented_csvdata['source_id'].isin(test_city_zone_list))
                ]
                
                # set the remaining rows for training and validation
                df_augmented_csvdata = df_augmented_csvdata.drop(df_test_city_zones.index)
                
                # append to test dataframe
                df_test = pd.concat([df_test, df_test_city_zones])
                
                # free up memory
                del df_test_city_zones
                gc.collect()
                
                # extract the rows from dataframe with matching hours of data for test
                df_test_hours_of_day = df_augmented_csvdata.loc[
                    df_augmented_csvdata['hour_of_day'].isin(
                        HYPER.TEST_SPLIT_DICT_UBERMOVEMENT['temporal_dict']['hours_of_day']
                    )
                ]
                
                # set the remaining rows for training and validation
                df_augmented_csvdata = df_augmented_csvdata.drop(df_test_hours_of_day.index)
                
                # append to test dataframe
                df_test = pd.concat([df_test, df_test_hours_of_day])
                
                # free up memory
                del df_test_hours_of_day
                gc.collect()
                
                # create training and validation datasets
                df_train_append = df_augmented_csvdata.sample(
                    frac=HYPER.TRAIN_VAL_SPLIT_UBERMOVEMENT,
                    random_state=HYPER.SEED
                )
                df_val_append = df_augmented_csvdata.drop(df_train_append.index)
                
                # free up memory     
                del df_augmented_csvdata   
                gc.collect()
                
                # append training dataset
                df_train = pd.concat([df_train, df_train_append])
                
                # free up memory     
                del df_train_append   
                gc.collect()
            
                # append validation dataset
                df_val = pd.concat([df_val, df_val_append])
                
                # free up memory     
                del df_val_append   
                gc.collect()
            
            
            ### Save resulting data in chunks
            df_train, train_chunk_counter = save_chunk(
                HYPER,
                df_train,
                train_chunk_counter,
                HYPER.PATH_TO_DATA_UBERMOVEMENT_TRAIN,
                'training_data'    
            )
            df_val, val_chunk_counter = save_chunk(
                HYPER,
                df_val,
                val_chunk_counter,
                HYPER.PATH_TO_DATA_UBERMOVEMENT_VAL,
                'validation_data'
            )
            df_test, test_chunk_counter = save_chunk(
                HYPER,
                df_test,
                test_chunk_counter,
                HYPER.PATH_TO_DATA_UBERMOVEMENT_TEST,
                'testing_data'
            )
            
        # update progress bar
        pbar.update(1)

    ### Tell us the rations that result from our splitting rules
    n_train = (train_chunk_counter * HYPER.CHUNK_SIZE_UBERMOVEMENT) + len(df_train.index)
    n_val = (val_chunk_counter * HYPER.CHUNK_SIZE_UBERMOVEMENT) + len(df_val.index)
    n_test = (test_chunk_counter * HYPER.CHUNK_SIZE_UBERMOVEMENT) + len(df_test.index)
    n_total = n_train + n_val + n_test
    
    print(
        "Training data   :    {:.0%} \n".format(n_train/n_total),
        "Validation data :    {:.0%} \n".format(n_val/n_total),
        "Testing data    :    {:.0%} \n".format(n_test/n_total)
    )
    
    ### Save results of last iteration
    df_train, train_chunk_counter = save_chunk(
        HYPER,
        df_train,
        train_chunk_counter,
        HYPER.PATH_TO_DATA_UBERMOVEMENT_TRAIN,
        'training_data',
        last_iteration=True  
    )
    df_val, val_chunk_counter = save_chunk(
        HYPER,
        df_val,
        val_chunk_counter,
        HYPER.PATH_TO_DATA_UBERMOVEMENT_VAL,
        'validation_data',
        last_iteration=True  
    )
    df_test, test_chunk_counter = save_chunk(
        HYPER,
        df_test,
        test_chunk_counter,
        HYPER.PATH_TO_DATA_UBERMOVEMENT_TEST,
        'testing_data',
        last_iteration=True  
    )
    
def save_chunk(
    config,
    df,
    chunk_counter,
    saving_path,
    filename,
    last_iteration=False 
):

    """ """
    
    ### Save resulting data in chunks
    while len(df.index) > HYPER.CHUNK_SIZE_UBERMOVEMENT or last_iteration:
        
        # increment chunk counter 
        chunk_counter += 1
        
        # create path
        path_to_saving = (
            saving_path
            + filename
            + '_{}.csv'.format(chunk_counter)
        )
        
        # shuffle
        df = df.sample(frac=1, random_state=HYPER.SEED)
        
        # save chunk
        df.iloc[:HYPER.CHUNK_SIZE_UBERMOVEMENT].to_csv(path_to_saving, index=False)
        
        # delete saved chunk
        if not last_iteration:
            df = df[HYPER.CHUNK_SIZE_UBERMOVEMENT:]
            
        # Must be set to exit loop on last iteration
        last_iteration = False
        
    return df, chunk_counter
    
