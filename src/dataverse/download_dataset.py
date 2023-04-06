import os
import json
import requests
import zipfile

def download(config: dict, dataset_name: str):
  """
  """
  print("Downloading data!")
  # set basic data and construct url
  with open(config['dataverse']['path_to_token'], 'r') as token:
    api_key = token.read().replace('\n', '')
  dataverse_server = config['dataverse']['base_url']
  persistentId = config['dataverse'][dataset_name]['persistentId']
  url_persistent_id = (
    "{}/api/access/dataset/:persistentId/?persistentId={}&key={}".format(
      dataverse_server, persistentId, api_key
    )
  )
  # set saving path
  saving_path = config['dataverse'][dataset_name]['saving_path']
  if not os.path.isdir(saving_path):
    os.mkdir(saving_path)
  saving_path += 'files.zip'
  """
  # download data
  r = requests.get(url_persistent_id)
  # write data
  with open(saving_path, 'wb') as file:
    file.write(r.content)
  """
  # unzip data
  with zipfile.ZipFile(saving_path, 'r') as zip:
    saving_path.replace('files.zip', '')
    zip.extractall(saving_path)
