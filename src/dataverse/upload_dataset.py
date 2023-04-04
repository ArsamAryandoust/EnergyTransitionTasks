import os
import json
import requests
import pandas as pd

def upload(config: dict, dataset_name: str):
  """
  """
  # get the base data for requested dataset
  dataverse_server = config['dataverse']['base_url']
  with open(config['dataverse']['path_to_token'], 'r') as token:
    api_key = token.read().replace('\n', '')
  persistentId = config['dataverse'][dataset_name]['persistentId']

  # set full path to dataset root directory
  if 'raw' in dataset_name:
    path_to_data = (
      config['general']['path_to_data_raw'] + dataset_name[:-4] + '/'
    )
  else:
    path_to_data = config['general']['path_to_data'] + dataset_name + '/'
  
  # iterate over all files in dataset directory
  base_path_len = len(path_to_data)
  """
  # if upload is interrupted, you can continue from chosen branch by un-commenting
  # and setting the path_to_data elsewhere below
  path_to_data += 'profiles_400/building-year profiles'
  """
  recursive_call(path_to_data, dataverse_server, persistentId, api_key, 
    base_path_len)

def recursive_call(path_to_dir, dataverse_server, persistentId, api_key,
  base_path_len):
  """
  """
  for entry in os.scandir(path_to_dir):
    if entry.is_file():
      save_file(entry, dataverse_server, persistentId, api_key, base_path_len)
    elif entry.is_dir():
      recursive_call(entry.path, dataverse_server, persistentId, api_key,
        base_path_len)

def save_file(entry, dataverse_server, persistentId, api_key, base_path_len):
  """
  """
  if '.csv' in entry.name:
    file_content = pd.read_csv(entry.path).to_csv(index=False)

  files = {'file': (entry.name, file_content)}
  path = entry.path[base_path_len:-len(entry.name)]
  params = {
    "tabIngest": "false",
    "directoryLabel": path
  }
  payload = {"jsonData": json.dumps(params)}
  url_persistent_id = (
    '{}/api/datasets/:persistentId/add?persistentId={}&key={}'.format(
      dataverse_server, persistentId, api_key
    )
  )
  try:
    r = requests.post(url_persistent_id, data=payload, files=files)
  except requests.exceptions.ConnectionError as e:
    r = "No response"

  """
  print('-' * 40)
  print(r.json())
  print(r.status_code)
  """

