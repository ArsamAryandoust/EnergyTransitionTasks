import selberai.data.load_data as load_data

def test(config: dict, name: str, subtask: str):
  """
  """
  print('Testing {}'.format(name))
  
  # set the path to data we want to load
  path_to_data = config['general']['path_to_data']+name+'/'+subtask+'/'    
  
  # import api_key
  with open(config['dataverse']['path_to_token'], 'r') as token:
    api_key = token.read().replace('\n', '')
  
  # load data
  dataset_dict = load_data.load(name, path_to_data=path_to_data, 
    token=api_key)
  
  
  
  
  print("Successfully tested {}!".format(name))
