


def analyse(config: dict, name: str, subtask: str):
  """
  """
  print('Analysing {} for subtask {}'.format(name, subtask))
  
  ###
  # Load data
  ###
  
  # set the path to data and token we want to load
  path_to_data = config['general']['path_to_data'] + name + '/' + subtask + '/'    
  path_to_token = config['dataverse']['path_to_token']
  
  # load data
  dataset = load_data.load(name, path_to_data=path_to_data, 
    path_to_token=path_to_token)
  
  
  ###
  # Analyse st-ood score
  ###
  
  ###
  # Analyse 
  ###
  
  
  
