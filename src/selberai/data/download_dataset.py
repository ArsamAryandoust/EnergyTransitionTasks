import os
import json
import requests
import shutil

def download(config: dict, dataset_name: str):
  """
  """
  
  # set saving path
  path_to_folder = config['dataverse'][dataset_name]['saving_path']
  path_to_file = path_to_folder + 'files.zip'
  if os.path.isdir(path_to_folder):
    # this check is needed when downloading processed data
    directory_content = os.listdir(path_to_folder)
    if len(directory_content) > 0:
      print("Directory is not empty. Check if data is not already downloaded!")
      exit(1)
  else:
    os.mkdir(path_to_folder)
    print("Downloading data!")
  
  # create requests session
  s = requests.Session()
  
  # set basic data and construct url
  with open(config['dataverse']['path_to_token'], 'r') as token:
    api_key = token.read().replace('\n', '')
  dataverse_server = config['dataverse']['base_url']
  persistentId = config['dataverse'][dataset_name]['persistentId']
  url_persistent_id = (
    "{}/api/access/dataset/:persistentId/?persistentId={}&key={}".format(
      dataverse_server, persistentId, api_key))
      
  # download data
  r = s.get(url_persistent_id)
  
  # write data
  with open(path_to_file, 'wb') as file:
    file.write(r.content)
    
  # unzip archive
  shutil.unpack_archive(path_to_file, path_to_folder)
  
  # remove .zip file
  os.remove(path_to_file)
  
