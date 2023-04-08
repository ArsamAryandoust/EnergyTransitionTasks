import selberai.data.load_data as load_data

def test(config: dict, dataset_name: str):
  """
  """
  
  # set the path to data we want to load
  path_to_data = config['general']['path_to_data']
  
  # load data
  load_data(dataset_name, path_to_data=path_to_data)
  
  
  print("Successfully tested {}!".format(dataset_name))
