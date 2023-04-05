import os
import json
import requests
import pandas as pd
import time

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
  
  # get the path base length for shortening paths later
  base_path_len = len(path_to_data)
  
  # load or set to empty a record of already uploaded files
  upload_fail_record = []
  
  # iterate over all files in dataset directory
  upload_fail_record = recursive_call(path_to_data, dataverse_server, 
    persistentId, api_key, base_path_len, upload_fail_record)

def recursive_call(path_to_dir: str, dataverse_server: str, persistentId: str, 
  api_key: str, base_path_len: int, upload_fail_record: dict) -> dict:
  """
  """
  for entry in os.scandir(path_to_dir):
    if entry.is_file():
      upload_fail_record = save_file(entry, dataverse_server, persistentId, api_key,        base_path_len, upload_fail_record)
    elif entry.is_dir():
      upload_fail_record = recursive_call(entry.path, dataverse_server, 
        persistentId, api_key, base_path_len, upload_fail_record)

  return upload_fail_record

def save_file(entry: os.DirEntry, dataverse_server: str, persistentId: str, 
  api_key: str, base_path_len: int, upload_fail_record: dict):
  """
  """
  if '.csv' in entry.name:
    file_content = pd.read_csv(entry.path).to_csv(index=False)
  elif '.json' in entry.name:
    file_content = json.dumps(json.load(entry.path))
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
  time.sleep(1)
  try:
    r = requests.post(url_persistent_id, data=payload, files=files)
  except requests.exceptions.ConnectionError as e:
    r = "No response"
    print(entry.name)
    print(entry.path)
    upload_fail_record.append((entry.path, entry.name))

  return upload_fail_record
