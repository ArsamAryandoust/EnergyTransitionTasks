


from config.load_oc import config_OC
from ocpmodels.datasets import TrajectoryLmdbDataset, SinglePointLmdbDataset


def process_all_datasets(config: dict, save: bool):
  """
  """
  print("Processing Open Catalyst dataset.")
  
  # iterate over all subtasks
  for subtask in config['OpenCatalyst']['subtask_list']:
  
    # augment conigurations with additional information
    config_opencat = config_OC(config, subtask, save)
    
    # do data processing according to current subtask
    if subtask == 'oc20_s2ef':
      process_oc20_s2ef(config_opencat, save)
      
    elif subtask == 'oc20_is2res':
      process_oc20_is2res(config_opencat, save)
      
    elif subtask == 'oc22_s2ef':
      process_oc22_s2ef(config_opencat, save)
      
    elif subtask == 'oc22_is2res':
      process_oc22_is2res(config_opencat, save)
      
    
    
    
def process_oc20_s2ef(config_opencat: dict, save: bool):
  """
  """
  
  pass
  
def process_oc20_is2res(config_opencat: dict, save: bool):
  """
  """
  
  pass
  
def process_oc22_s2ef(config_opencat: dict, save: bool):
  """
  """
  
  pass
  
def process_oc22_is2res(config_opencat: dict, save: bool):
  """
  """
  
  pass
