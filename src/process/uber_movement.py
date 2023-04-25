import math
import os
import gc
import random
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

from load_config import config_UM


def process_all_datasets(config: dict, save:bool):
  """
  Processes all datasets for Uber Movement prediction task.
  """
  print("\nProcessing Uber Movement dataset.")
  
  # iterated over all subtasks
  for subtask in config['uber_movement']['subtask_list']:
  
    # augment conigurations with additional information
    config_uber = config_UM(config, subtask, save)
        
    # process geographic information
    cityzone_centroid_df_dict = process_geographic_information(config_uber)
    
    # transform coordinates
    cityzone_centroid_df_dict = transform_coordinates(cityzone_centroid_df_dict)
    
    # create and save train, val and test datasets
    create_train_val_test(config_uber, cityzone_centroid_df_dict, save)
    
    
def create_train_val_test(config_uber: dict, cityzone_centroid_df_dict: dict, 
  save: bool):
  """
  """
  # create new dataframes and chunk counters here
  (df_train, df_val, df_test, train_file_count, val_file_count, 
    test_file_count) = load_df_and_file_counters(config_uber)
    
    
def augment_csv(config_uber: dict, df_csv_dict: dict, 
  cityzone_centroid_df_dict: dict, city: str):
  """ 
  Augments data points of df with city id, year, quarter of yeear and daytype
  information.
  """
  
  # copy centroid dict
  centroid_dict = cityzone_centroid_df_dict[city]
  
  # copy raw dataframe
  df_augmented = df_csv_dict['df']
  
  # subsample or shuffle data (for frac=1)    
  df_augmented = df_augmented.sample(frac=config_uber['subsample_frac'],
    random_state=config_uber['seed'])
  
  # augment raw dataframe
  df_augmented.insert(0, 'city_id', config_uber['city_id_mapping'][city])
  df_augmented.insert(3, 'year', df_csv_dict['year'])
  df_augmented.insert(4, 'quarter_of_year', df_csv_dict['quarter_of_year'])
  df_augmented.insert(6, 'daytype', df_csv_dict['daytype'])
  
  # rename some columns with better names
  df_augmented.rename(columns={'hod':'hour_of_day', 'sourceid':'source_id', 
    'dstid':'destination_id'}, inplace=True)
  
  # remove any rows with nan entry
  df_augmented = df_augmented[df_augmented.isnull().sum(axis=1) < 1]

  ### Map source ID coordinates ###
  # rename columns
  centroid_dict_new = centroid_dict.rename(
    columns={'city_zone': 'source_id', 'x': 'x_source', 'y': 'y_source', 
    'z': 'z_source'}
  )
  
  # merge columns
  df_augmented = df_augmented.merge(centroid_dict_new, on='source_id')
  
  ### Map source ID coordinates ###
  # rename columns
  centroid_dict_new = centroid_dict.rename(
    columns={'city_zone': 'destination_id', 'x': 'x_dest', 'y': 'y_dest', 
    'z': 'z_dest'}
  )
  
  # merge columns
  df_augmented = df_augmented.merge(centroid_dict_new, on='destination_id')
  
  # rearrange column names
  col_list = df_augmented.columns.to_list()
  col_list.remove('x_source'), col_list.remove('y_source') 
  col_list.remove('z_source'), col_list.remove('x_dest'), 
  col_list.remove('y_dest'), col_list.remove('z_dest')

  col_list.insert(1, 'x_source'), col_list.insert(2, 'y_source')
  col_list.insert(3, 'z_source'), col_list.insert(4, 'x_dest')
  col_list.insert(5, 'y_dest'), col_list.insert(6, 'z_dest')
  
  df_augmented = df_augmented[col_list]
  
  return df_augmented    
    
    
def import_csvdata(config_uber: dict, city: str):
    """ 
    Imports the Uber Movement data for a passed city 
    """
    
    # get the files dictionary and create an empty list to fill dataframes of csv
    df_csv_dict_list = []
    
    # iterate over all csv files of current city
    for csv_file_dict in config_uber['csv_file_dict_list'][city]:
        
        # set the path to currently iterated csv file of city
        path_to_csv = (config_uber['path_to_data_raw'] + city + '/' 
            + csv_file_dict['filename'])
        
        # import csv data as pandas dataframe
        df_csv = pd.read_csv(path_to_csv)
        
        # create a copy of csv dataframe dict and append new csv dataframe as df
        csv_df_dict = csv_file_dict.copy()
        csv_df_dict['df'] = df_csv
        df_csv_dict_list.append(csv_df_dict)
        
    return df_csv_dict_list
  
  
  
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
  
  
def transform_coordinates(cityzone_centroid_df_dict: dict) -> (dict):
  """
  """
  
  # iterate over all city, dataframe pairs
  for city, dataframe in cityzone_centroid_df_dict.items():
  
    # calculate values you need for 
    df_sin_lon = dataframe.lon.map(sin_transform)
    df_sin_lat = dataframe.lat.map(sin_transform)
    df_cos_lat = dataframe.lat.map(cos_transform)
    df_cos_lon = dataframe.lon.map(cos_transform)
    
    # calculate coordaintes
    df_x_cord = df_cos_lat.mul(df_cos_lon)
    df_y_cord = df_cos_lat.mul(df_sin_lon)
    df_z_cord = df_sin_lat
    
    # get city zone column
    cityzone = dataframe.city_zone
    
    # reaplce old dataframe with a new dataframe
    cityzone_centroid_df_dict[city] = pd.concat(
      [cityzone, df_x_cord, df_y_cord, df_z_cord], axis=1)
    
    # set desired column names
    cityzone_centroid_df_dict[city].columns = ['city_zone','x', 'y', 'z']
    
    
  return cityzone_centroid_df_dict
    
    
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
    
    
def process_geographic_information(config_uber: dict):
  """
  """
  print("Processing geographic information.")
  
  # create progress bar
  pbar = tqdm(total=len(config_uber['list_of_cities']))
  
  # declare results dict we want to return
  cityzone_centroid_df_dict = {}
  
  # iterate over list of cities in current subtask
  for city in config_uber['list_of_cities']:
  
    # import geojson for iterated city
    geojson_dict = import_geojson(config_uber, city)
  
    # get only features entry
    geojson_dict = geojson_dict['features']

    # create a mapping of json entries to uber movement city zone ids
    map_json_entry_to_movement_id = dict()
    for json_id, json_entry in enumerate(geojson_dict):
      map_json_entry_to_movement_id[json_id] = int(
        json_entry['properties']['MOVEMENT_ID'])
      
    # create mappings of movement ids to empty list for lat/long coordinates
    map_movement_id_to_coordinates = {
      'lat' : dict(),
      'lon' : dict()
    }
    for k, v in map_json_entry_to_movement_id.items():
        map_movement_id_to_coordinates['lat'][v] = []
        map_movement_id_to_coordinates['lon'][v] = []
      
    # iterate over all movement IDs and json IDs to get coordinates and flatten
    for json_id, movement_id in map_json_entry_to_movement_id.items():
      coordinates = geojson_dict[json_id]['geometry']['coordinates']
      map_movement_id_to_coordinates = foster_coordinates_recursive(movement_id, 
        map_movement_id_to_coordinates, coordinates)
      
    # calculate centroids of city zone polygons
    map_movement_id_to_centroids = calc_centroids(
      map_movement_id_to_coordinates)
    
    
    # create dataframes for lats and longs of each city zone
    df_latitudes = pd.DataFrame.from_dict(
      map_movement_id_to_centroids['lat'], orient='index', columns=['lat'])
    df_longitudes = pd.DataFrame.from_dict(
      map_movement_id_to_centroids['lon'], orient='index', columns=['lon'])

    # merge dataframes
    df_centroids = pd.concat([df_latitudes, df_longitudes], axis=1)

    # index is city zone. Add these as their own column
    df_centroids['city_zone'] = df_centroids.index
    
    # add to results dictionary
    cityzone_centroid_df_dict[city] = df_centroids
    
    # update progress bar
    pbar.update(1)
    
  return cityzone_centroid_df_dict    


def import_geojson(config_uber: dict, city: str) -> dict:
  """ 
  Uses the city to file mapping of city to load the geo-json file and returns
  it as a dicitonary.
  """
  
  # set filename
  filename = config_uber['json'][city]
  
  # set path
  path_to_json = config_uber['path_to_data_raw'] + city + '/' + filename
  
  # load data
  with open(path_to_json, 'r') as json_file:
    geojson_dict = json.load(json_file)
  
  return geojson_dict


    
def foster_coordinates_recursive(movement_id: int,
  map_movement_id_to_coordinates: dict,
  coordinates: pd.Series) -> (dict):
  """ Flattens the coordinates of a passed city zone id (movement_id)
  and coordiates list recursively and saves their numeric values
  in the dictionaries that map movement ids to a list of latitude and 
  longitude coordinates.
  """
  
  dummy = 0
  for j in coordinates:
  
    if type(j) != list and dummy == 0:
      map_movement_id_to_coordinates['lon'][movement_id].append(j)
      dummy = 1
      continue
      
    elif type(j) != list and dummy == 1:
      map_movement_id_to_coordinates['lat'][movement_id].append(j)
      break
      
    else:
      dummy = 0
      coordinates = j
      map_movement_id_to_coordinates = foster_coordinates_recursive(movement_id,
        map_movement_id_to_coordinates, coordinates)
    
  return map_movement_id_to_coordinates


def calc_centroids(map_movement_id_to_coordinates: dict) -> (dict):
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
    map_movement_id_to_coordinates['lat'].items()):
    # get also the longitude coordinates
    long_coordinates = map_movement_id_to_coordinates['lon'][movement_id]
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
    map_movement_id_to_coordinates['lat'].items()):
    # get also the longitude coordinates
    long_coordinates = map_movement_id_to_coordinates['lon'][movement_id]
    
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
    
  map_movement_id_to_centroids = {
    'lat': map_movement_id_to_centroid_lat,
    'lon': map_movement_id_to_centroid_long
  }
      
  return map_movement_id_to_centroids
