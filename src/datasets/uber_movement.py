import math
import os
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
    print("\nProcessing Uber Movement dataset.")
    # iterated over all subtasks
    for subtask in config['uber_movement']['subtask_list']:
        # augment conigurations with additional information
        config_uber = config_UM(config, subtask)
        # process geographic information
        city_zone_shift_dict = process_geographic_information(config_uber)
        # split training validation testing
        split_train_val_test(config_uber, city_zone_shift_dict)


def process_geographic_information(config_uber: dict) -> dict:
    """
    Processes and saves geographic features of cities and their zones.
    """
    print('\nProcessing geographic data.')
    # create progress bar
    pbar = tqdm(total=len(config_uber['list_of_cities']))
    # create empty dict to record city zone shifting factor for making every
    # city zone start from ID number 1.
    city_zone_shift_dict = {}
    # declare minimum and maxmimum number of edges of ciy zone coordinates we want
    # to calculate
    min_polygon_edges, max_polygon_edges = 1e3, 0
    # iterate over all cities
    for city in config_uber['list_of_cities']:
        # import geojson for iterated city
        df_geojson = import_geojson(config_uber, city)
        # extract geojson information of city zones as latitude and longitude df
        df_latitudes, df_longitudes = process_geojson(df_geojson)
        # do the shifting here
        shift_factor_add = 1 - min(df_latitudes.columns)
        city_zone_shift_dict[city] = shift_factor_add
        if shift_factor_add != 0:
            print("\n Shifting factor {} is used for geographic data {}".format(
                shift_factor_add, city))
            lat_col = [x+shift_factor_add for x in df_latitudes.columns.to_list()]
            long_col = [x+shift_factor_add for x in df_longitudes.columns.to_list()]
            df_latitudes.columns = lat_col
            df_longitudes.columns = long_col
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
        df_geographic_info = pd.concat([df_x_cord, df_y_cord, df_z_cord], axis=1)
        filename = city + '.csv'
        saving_path = config_uber['path_to_data_add'] + filename
        df_geographic_info.to_csv(saving_path)
        ### Calculate minimum and maximum of city zone polygon edges ###
        # iterate over all columns of dataframe
        for col in df_z_cord.columns:
            # get size of column
            len_col = len(df_z_cord[col].dropna())
            # update minimum and maximum if relevant
            if len_col < min_polygon_edges:
                min_polygon_edges = len_col
            if len_col > max_polygon_edges:
                max_polygon_edges = len_col
        # update progress bar
        pbar.update(1)
    # print out miminimum and maximum of city zone polygon edges here
    print("\nMinimum of city zone polygon edges: {}".format(min_polygon_edges))
    print("\nMaximum of city zone polygon edges: {}".format(max_polygon_edges))
    return city_zone_shift_dict

def import_geojson(config_uber: dict, city: str) -> pd.DataFrame:
    """ 
    Uses the city to file mapping of city to load the geo-json file and returns
    it as a dataframe.
    """
    filename = config_uber['json'][city]
    path_to_json = config_uber['path_to_data_raw'] + city + '/' + filename
    df_geojson = pd.read_json(path_to_json)
    return df_geojson


def process_geojson(df_geojson: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """ Maps Uber Movement city zone IDs to a flattened list of latitude and 
    longitude coordinates in the format of two dictionaries. Uses the recursive 
    function called foster_coordinates_recursive to flatten the differently 
    nested data.
    """
    df_geojson.pop('type')
    df_geojson = df_geojson['features']
    # create a mapping of json entries to uber movement city zone ids
    map_json_entry_to_movement_id = dict()
    for json_id, json_entry in enumerate(df_geojson):
        map_json_entry_to_movement_id[json_id] = int(
          json_entry['properties']['MOVEMENT_ID'])
    # create a mappings of movement ids to empty list for lat and long coordinates
    map_movement_id_to_latitude_coordinates = dict()
    map_movement_id_to_longitude_coordinates = dict()
    for k, v in map_json_entry_to_movement_id.items():
        map_movement_id_to_latitude_coordinates[v] = []
        map_movement_id_to_longitude_coordinates[v] = []
    # iterate over all movement IDs and json IDs to get coordinates and flatten
    for json_id, movement_id in map_json_entry_to_movement_id.items():
        coordinates = df_geojson[json_id]['geometry']['coordinates']
        (map_movement_id_to_latitude_coordinates, 
            map_movement_id_to_longitude_coordinates
        ) = foster_coordinates_recursive(movement_id, 
            map_movement_id_to_latitude_coordinates,
            map_movement_id_to_longitude_coordinates, coordinates)
    # calculate centroids of city zone polygons
    (map_movement_id_to_centroid_lat, map_movement_id_to_centroid_long
    ) = calc_centroids(map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates)
    # add centroid coordinates to beginning of dictionary lists
    for k, v in map_json_entry_to_movement_id.items():
        map_movement_id_to_latitude_coordinates[v].insert(0, 
            map_movement_id_to_centroid_lat[v])
        map_movement_id_to_longitude_coordinates[v].insert(0, 
            map_movement_id_to_centroid_long[v])
    # create dataframes for lats and longs of each city zone
    df_latitudes = pd.DataFrame.from_dict(
        map_movement_id_to_latitude_coordinates, orient='index').transpose()
    df_longitudes = pd.DataFrame.from_dict(
        map_movement_id_to_longitude_coordinates, orient='index').transpose()
    return df_latitudes, df_longitudes
  
  
def foster_coordinates_recursive(movement_id: int,
    map_movement_id_to_latitude_coordinates: dict,
    map_movement_id_to_longitude_coordinates: dict,
    coordinates: pd.Series) -> (dict, dict):
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
            (map_movement_id_to_latitude_coordinates,
                map_movement_id_to_longitude_coordinates
            ) = foster_coordinates_recursive(movement_id,
                map_movement_id_to_latitude_coordinates,
                map_movement_id_to_longitude_coordinates, coordinates)
    map_movement_id_to_coordinates = (
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates)
    return map_movement_id_to_coordinates


def calc_centroids(map_movement_id_to_latitude_coordinates: dict,
    map_movement_id_to_longitude_coordinates: dict) -> (dict, dict):
    """ Calculates the centroid of passed city zone polygons. Should a city
    zone consist of unregularities or multiple polygons, this is identified
    by centroid coordinates that are not within the bound of minimum and 
    maximum values of all coordinates of that city zone. In this case, the
    centroids are replaced with the mean of lat and long coordinates.
    """
    # create empty dictionary for mapping Uber Movement IDs to city zone areas
    map_movement_id_to_cityzone_area = dict()
    # iterate over all movement IDs and latitude coordinates
    for movement_id, lat_coordinates in (
        map_movement_id_to_latitude_coordinates.items()):
        # get also the longitude coordinates
        long_coordinates = map_movement_id_to_longitude_coordinates[movement_id]
        # calculate currently iterated city zone area
        area_cityzone = 0
        for i in range(len(lat_coordinates)-1):
            area_cityzone = (area_cityzone
                + long_coordinates[i] * lat_coordinates[i+1]
                - long_coordinates[i+1] * lat_coordinates[i])
        area_cityzone = (area_cityzone
            + long_coordinates[i+1] * lat_coordinates[0]
            - long_coordinates[0] * lat_coordinates[i+1])
        area_cityzone *= 0.5
        map_movement_id_to_cityzone_area[movement_id] = area_cityzone
    # create empty dictionaries for mapping Uber Movement IDs to city zone centroids
    map_movement_id_to_centroid_lat = dict()
    map_movement_id_to_centroid_long = dict()
    # iterate over all movement IDs and latitude coordinates
    for movement_id, lat_coordinates in (
        map_movement_id_to_latitude_coordinates.items()):
        # get also the longitude coordinates
        long_coordinates = map_movement_id_to_longitude_coordinates[movement_id]
        # calculate currently iterated city zone area
        centroid_lat = 0
        centroid_long = 0
        for i in range(len(lat_coordinates)-1):
            centroid_long += (long_coordinates[i]+ long_coordinates[i+1]) * (
                long_coordinates[i] * lat_coordinates[i+1]
                - long_coordinates[i+1] * lat_coordinates[i])
            centroid_lat += (lat_coordinates[i] + lat_coordinates[i+1]) * (
                long_coordinates[i] * lat_coordinates[i+1]
                - long_coordinates[i+1] * lat_coordinates[i])
        centroid_long += (long_coordinates[i+1] + long_coordinates[0]) * (
            long_coordinates[i+1] * lat_coordinates[0]
            - long_coordinates[0] * lat_coordinates[i+1])
        centroid_lat += (lat_coordinates[i+1] + lat_coordinates[0]) * (
                long_coordinates[i+1] * lat_coordinates[0]
                - long_coordinates[0] * lat_coordinates[i+1])
        centroid_lat /= 6 * map_movement_id_to_cityzone_area[movement_id]
        centroid_long /=  6 * map_movement_id_to_cityzone_area[movement_id]
     
        # Uber Movement city zones sometimes consist of multiple distinct polygons
        if (centroid_lat < min(lat_coordinates)
            or centroid_lat > max(lat_coordinates)
            or centroid_long < min(long_coordinates)
            or centroid_long > max(long_coordinates)):
            # in this case we calculate the mean instead of centroid
            centroid_lat = np.mean(lat_coordinates)
            centroid_long = np.mean(long_coordinates)            
        map_movement_id_to_centroid_lat[movement_id] = centroid_lat
        map_movement_id_to_centroid_long[movement_id] = centroid_long
    map_movement_id_to_centroid_coordinates = (map_movement_id_to_centroid_lat,
        map_movement_id_to_centroid_long)
    return map_movement_id_to_centroid_coordinates
  
    
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
    

def load_df_and_file_counters(config_uber: dict) -> (pd.DataFrame, pd.DataFrame, 
    pd.DataFrame, int, int, int):
    """
    Loads the last file that was saved from previous subtask as dataframe and
    sets the file counters accordingly.
    """
    # decleare empty dataframes for trainining validation and testing
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    if config_uber['subtask'] == 'cities_10':
        # declare data point counters as zero
        train_file_count, val_file_count, test_file_count = 1, 1, 1
    else:
        # declare data point counters
        train_file_count = len(os.listdir(config_uber['path_to_data_train']))
        val_file_count = len(os.listdir(config_uber['path_to_data_val']))
        test_file_count = len(os.listdir(config_uber['path_to_data_test']))
        # load last datframes
        if train_file_count == 0:
            train_file_count = 1
        else:
            loading_path = (config_uber['path_to_data_train'] + 
                'training_data_{}.csv'.format(train_file_count))
            df_train = pd.read_csv(loading_path)
            
        if val_file_count == 0:
            val_file_count = 1
        else:
            loading_path = (config_uber['path_to_data_val'] + 
                'validation_data_{}.csv'.format(val_file_count))
            df_val = pd.read_csv(loading_path)
            
        if test_file_count == 0:
            test_file_count = 1
        else:
            loading_path = (config_uber['path_to_data_test'] + 
                'testing_data_{}.csv'.format(test_file_count))
            df_test = pd.read_csv(loading_path)
    # set return values
    return_values = (df_train, df_val, df_test, train_file_count, val_file_count, 
        test_file_count)
    return return_values   
        
        
def split_train_val_test(config_uber: dict, city_zone_shift_dict: dict):
    """ 
    Splits and saves datasets according to configuration rules.
    """
    # create new dataframes and chunk counters here
    (df_train, df_val, df_test, train_file_count, val_file_count, 
        test_file_count) = load_df_and_file_counters(config_uber)
    # iterate over all available cities
    for city in config_uber['list_of_cities']:
        print('\nProcessing data for:', city)
        # get city zone shifting factor for current city
        shift_factor_add = city_zone_shift_dict[city]
        # check if city is in testing city list
        if city in config_uber['spatial_ood']['ood_cities']:
            testing_city = True
        else:
            testing_city = False
        # import all csv files for currently iterated city
        df_csv_dict_list = import_csvdata(config_uber, city, shift_factor_add)
        # create progress bar
        pbar = tqdm(total=len(df_csv_dict_list))
        # iterate over all imported csv files for this city
        first_iteration = True
        for df_csv_dict in df_csv_dict_list:
            # check if testing year
            if df_csv_dict['year'] in config_uber['temporal_ood']['ood_years']:
                testing_year = True
            else:
                testing_year = False
            # check if testing quarter of year
            if df_csv_dict['quarter_of_year'] in (
                config_uber['temporal_dict']['ood_quarters_of_year']):
                testing_quarter = True
            else:
                testing_quarter = False
            # augment csv
            df_augmented = process_csvdata(config_uber, df_csv_dict, city)
            # free up memory     
            del df_csv_dict['df']
            gc.collect()
            # get the subset of city zones for test splits once per city
            if first_iteration:
                n_city_zones = max(df_augmented['source_id'].max(),
                    df_augmented['destination_id'].max())
                # get number of test city zones you want to split
                n_test_city_zones = round(
                    n_city_zones * config_uber['spatial_test_split'])
                # randomly sample test city zones
                random.seed(config_uber['seed'])
                test_city_zone_list = random.sample(range(n_city_zones), 
                    n_test_city_zones)
                # set false so as to not enter branch anymore
                first_iteration= False
            if testing_city or testing_year or testing_quarter:
                # append all data to test dataframe
                df_test = pd.concat([df_test, df_augmented], ignore_index=True)
                # free up memory     
                del df_augmented   
                gc.collect()
            else:
                # extract rows from dataframe with matching city zones
                df_test_city_zones = df_augmented.loc[
                    (df_augmented['destination_id'].isin(test_city_zone_list)) 
                    | (df_augmented['source_id'].isin(test_city_zone_list))]
                # set the remaining rows for training and validation
                df_augmented = df_augmented.drop(df_test_city_zones.index)
                # append to test dataframe
                df_test = pd.concat([df_test, df_test_city_zones], ignore_index=True)
                # free up memory
                del df_test_city_zones
                gc.collect()
                # extract the rows from dataframe with matching hours of day for test
                df_test_hours_of_day = df_augmented.loc[
                    df_augmented['hour_of_day'].isin(
                        config_uber['temporal_dict']['hours_of_day'])]
                # set the remaining rows for training and validation
                df_augmented = df_augmented.drop(df_test_hours_of_day.index)
                # append to test dataframe
                df_test = pd.concat([df_test, df_test_hours_of_day], ignore_index=True)
                # free up memory
                del df_test_hours_of_day
                gc.collect()
                # append remaining data to training dataset
                df_train = pd.concat([df_train, df_augmented], ignore_index=True)
                # free up memory     
                del df_augmented   
                gc.collect()
            # this condition guarantees validation splits at good moments
            if (len(df_test) * (1 - config_uber['val_test_split'])
                > config_uber['datapoints_per_file']):
                # split off validation data from ood testing data
                df_val_append = df_test.sample(
                    frac=config_uber['val_test_split'], 
                    random_state=config_uber['seed'])
                # remove validation data from test
                df_test = df_test.drop(df_val_append.index)
                # append to validation dataframe
                df_val = pd.concat([df_val, df_val_append], ignore_index=True)
                # free up memory     
                del df_val_append   
                gc.collect()
                # save testing and validation data chunks
                df_test, test_file_count = save_chunk(config_uber, df_test, 
                    test_file_count, config_uber['path_to_data_test'], 
                    'testing_data')
                df_val, val_file_count = save_chunk(config_uber, df_val,
                    val_file_count, config_uber['path_to_data_val'], 
                    'validation_data')
            # save training data chunks
            df_train, train_file_count = save_chunk(config_uber, df_train,
                train_file_count, config_uber['path_to_data_train'], 
                'training_data')
            # update progress bar
            pbar.update(1)
    ### Tell us the ratios that result from our splitting rules
    n_train = (train_file_count * config_uber['datapoints_per_file'] 
        + len(df_train))
    n_val = (val_file_count * config_uber['datapoints_per_file'] 
        + len(df_val))
    n_test = (test_file_count * config_uber['datapoints_per_file'] 
        + len(df_test))
    n_total = n_train + n_val + n_test
    print("Training data   :   {}/{} {:.0%}".format(n_train, n_total, 
            n_train/n_total),
        "\nValidation data :   {}/{} {:.0%}".format(n_val, n_total,
            n_val/n_total),
        "\nTesting data    :   {}/{} {:.0%}".format(n_test, n_total,
            n_test/n_total))
    
    ### Save results of last iteration
    df_train, train_file_count = save_chunk(config_uber, df_train, 
        train_file_count, config_uber['path_to_data_train'], 'training_data', 
        last_iteration=True)
    df_val, val_file_count = save_chunk(config_uber, df_val, val_file_count,
        config_uber['path_to_data_val'], 'validation_data', last_iteration=True)
    df_test, test_file_count = save_chunk(config_uber, df_test, test_file_count,
        config_uber['path_to_data_test'], 'testing_data', last_iteration=True)
    
    
def import_csvdata(config_uber: dict, city: str, shift_factor_add: int):
    """ 
    Imports the Uber Movement data for a passed city 
    """
    # get the files dictionary and create an empty list to fill dataframes of csv
    df_csv_dict_list = []
    # iterate over all csv files of current city
    for csv_file_dict in config_uber['csv_file_dict_list'][city]:
        # set the path to currently iterated csv ile of city
        path_to_csv = (config_uber['path_to_data_raw'] + city + '/' 
            + csv_file_dict['filename'])
        # import csv data as pandas dataframe
        df_csv = pd.read_csv(path_to_csv)
        # shift city zone IDs in case the shift factor is not zero. Note that 
        # during the the processing lat and long data, this has been done too.
        if shift_factor_add != 0:
            print("\nShifting factor {} is used for csv data of {}".format(
                shift_factor_add, city))
            df_csv['sourceid'] += 1
            df_csv['dstid'] += 1
        # create a copy of csv dataframe dict and append new csv dataframe as df
        csv_df_dict = csv_file_dict.copy()
        csv_df_dict['df'] = df_csv
        df_csv_dict_list.append(csv_df_dict)
    return df_csv_dict_list
    
    
def process_csvdata(config_uber: dict, df_csv_dict: pd.DataFrame, city: str):
    """ 
    Augments data points of df with city id, year, quarter of yeear and daytype
    information.
    """
    # copy raw dataframe
    df_augmented = df_csv_dict['df']
    
    # subsample or shuffle data (for frac=1)    
    df_augmented = df_augmented.sample(frac=config_uber['subsample_frac'],
        random_state=config_uber['seed'])
    # augment raw dataframe
    df_augmented.insert(0, 'city_id', config_uber['city_id_mapping'][city])
    df_augmented.insert(3, 'year', df_csv_dict['year'])
    df_augmented.insert(4, 'quarter_of_year', df_csv_dict['quarter_of_year'])
    df_augmented.insert(5, 'daytype', df_csv_dict['daytype'])
    # rename some columns with more clear names
    df_augmented.rename(columns={'hod':'hour_of_day', 'sourceid':'source_id', 
            'dstid':'destination_id'}, inplace=True)
    # remove any rows with nan entry
    df_augmented = df_augmented[df_augmented.isnull().sum(axis=1) < 1]
    return df_augmented
    
    
def save_chunk(config_uber: dict, df: pd.DataFrame, chunk_counter: int, 
    saving_path: str, filename: str, last_iteration=False) -> (pd.DataFrame, int):
    """ 
    Save a chunk of data and return remaining with chunk counter 
    """
    while (len(df) > config_uber['datapoints_per_file'] or last_iteration):
        # create path to saving
        path_to_saving = saving_path + filename + '_{}.csv'.format(chunk_counter)
        # shuffle dataframe
        df = df.sample(frac=1, random_state=config_uber['seed'])
        # save chunk
        if len(df) > 0:
            df.iloc[:config_uber['datapoints_per_file']].to_csv(path_to_saving, 
                index=False)
        # delete saved chunk
        df = df[config_uber['datapoints_per_file']:]
        # Must be set to exit loop on last iteration
        last_iteration = False
        # increment chunk counter 
        chunk_counter += 1
    return df, chunk_counter
    
    
    
    
