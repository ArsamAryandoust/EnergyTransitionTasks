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
