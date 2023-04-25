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
  
  # create progress bar
  pbar = tqdm(total=len(config_uber['list_of_cities']))
  
  # declare results dict we want to return
  cityzone_centroid_df_dict = {}
  
  # iterate over list of cities in current subtask
  for city in config_uber['list_of_cities']:
  
    # import geojson for iterated city
    geojson_dict = import_geojson(config_uber, city)
  
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


    
    
