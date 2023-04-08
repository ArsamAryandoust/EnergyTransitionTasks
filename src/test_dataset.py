import selberai.data.load_data as load_data

def test(config: dict, name: str):
  """
  """
  print('Testing {}'.format(name))
  
  # iterate over all subtasks
  for subtask in config[name]['subtask_list']:
    
    # set the path to data we want to load
    path_to_data = config['general']['path_to_data']+name+'/'+subtask+'/'    
    
    # load data
    train, val, test = load_data.load(dataset_name, path_to_data=path_to_data)
  
  
  
  print("Successfully tested {}!".format(name))
