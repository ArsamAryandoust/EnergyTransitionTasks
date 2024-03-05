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
  if '_raw' in dataset_name:
    path_to_data = (
      config['general']['path_to_data_raw'] + dataset_name[:-4] + '/'
    )
  else:
    path_to_data = config['general']['path_to_data'] + dataset_name + '/'
  
  # get the path base length for shortening paths later
  base_path_len = len(path_to_data)
  
  # load or set to empty a record of failed upload files
  upload_fail_record = []
  
  # create requests session
  s = requests.Session()
  
  # iterate over all files in dataset directory
  upload_fail_record = recursive_call(path_to_data, dataverse_server, 
    persistentId, api_key, base_path_len, upload_fail_record, s)
  
  # save upload fail record as json
  upload_fail_record = json.dumps(dict(upload_fail_record))
  path_to_record = '.upload_records/'
  if not os.path.isdir(path_to_record):
    os.mkdir(path_to_record)
  path_to_record += dataset_name + '.json'
  with open(path_to_record, 'w') as write_file:
    json.dump(upload_fail_record, write_file)


def recursive_call(path_to_dir: str, dataverse_server: str, persistentId: str, 
  api_key: str, base_path_len: int, upload_fail_record: dict, s) -> dict:
  """
  """
  
  for entry in os.scandir(path_to_dir):
    if entry.is_file():
      upload_file(entry.name, entry.path, dataverse_server, persistentId,
        api_key, base_path_len, s)
    elif entry.is_dir():
      try: 
        upload_fail_record = recursive_call(entry.path, dataverse_server, 
          persistentId, api_key, base_path_len, upload_fail_record, s)
      except:
        print("Exception occurred!\n", entry.path)
        upload_fail_record.append((entry.path, entry.name))
        
  return upload_fail_record


def upload_file(entry_name: str, entry_path: str, dataverse_server: str, 
  persistentId: str, api_key: str, base_path_len: int, s):
  """
  """
  
  if ('.csv' in entry_name or '.CSV' in entry_name):
    file_content = pd.read_csv(entry_path).to_csv(index=False)
    
  elif '.json' in entry_name:
    with open(entry_path, 'r' ) as file_json:
      file_content = json.dumps(json.load(file_json))
    
  files = {'file': (entry_name, file_content)}
  path = entry_path[base_path_len:-len(entry_name)]
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
  s.post(url_persistent_id, data=payload, files=files)



