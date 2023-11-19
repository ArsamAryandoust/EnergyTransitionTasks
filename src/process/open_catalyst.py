import os
import gc
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import ocpmodels
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
    
    # create meta data
    _, _, _, _ = create_add_data(config_opencat, save)
    
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
  print('\nProcessing OC20_S2EF.\n')
  
  # create all relevant data paths
  p_train = config_opencat['path_to_data_raw_oc20_s2ef_train']
  p_val_id = config_opencat['path_to_data_raw_oc20_s2ef_val_id']
  p_val_ood_ads = config_opencat['path_to_data_raw_oc20_s2ef_val_ood_ads']
  p_val_ood_cat = config_opencat['path_to_data_raw_oc20_s2ef_val_ood_cat']
  p_val_ood_both = config_opencat['path_to_data_raw_oc20_s2ef_val_ood_both']
  
  # create train, val, test data
  create_s2ef_data(config_opencat, 
    [p_train, p_val_id, p_val_ood_ads, p_val_ood_cat, p_val_ood_both], 
    save=save)
  
  
def process_oc20_is2res(config_opencat: dict, save: bool):
  """
  """
  print('\nProcessing OC20_IS2RES.\n')
  
  # create all relevant data paths
  p_train = config_opencat['path_to_data_raw_oc20_is2res_train']
  p_val_id = config_opencat['path_to_data_raw_oc20_is2res_val_id']
  p_val_ood_ads = config_opencat['path_to_data_raw_oc20_is2res_val_ood_ads']
  p_val_ood_cat = config_opencat['path_to_data_raw_oc20_is2res_val_ood_cat']
  p_val_ood_both = config_opencat['path_to_data_raw_oc20_is2res_val_ood_both']

  # create train, val, test data
  create_is2res_data(config_opencat, 
    [p_train, p_val_id, p_val_ood_ads, p_val_ood_cat, p_val_ood_both], 
    save=save, single_lmdb=True)
  
  
  
def process_oc22_s2ef(config_opencat: dict, save: bool):
  """
  """
  print('\nProcessing OC22_S2EF.\n')
  
  # create all relevant data paths
  p_train = config_opencat['path_to_data_raw_oc22_s2ef_train']
  p_val_id = config_opencat['path_to_data_raw_oc22_s2ef_val_id']
  p_val_ood = config_opencat['path_to_data_raw_oc22_s2ef_val_ood']

  # create train, val, test data
  create_s2ef_data(config_opencat, [p_train, p_val_id, p_val_ood], save=save)
  
  
  
def process_oc22_is2res(config_opencat: dict, save: bool):
  """
  """
  print('\nProcessing OC22_IS2RES.\n')
  
  # create all relevant data paths
  p_train = config_opencat['path_to_data_raw_oc22_is2res_train']
  p_val_id = config_opencat['path_to_data_raw_oc22_is2res_val_id']
  p_val_ood = config_opencat['path_to_data_raw_oc22_is2res_val_ood']
  
  # Create training data from all in distribution datasets
  create_is2res_data(config_opencat, [p_train, p_val_id, p_val_ood], save=save)
  


def create_add_data(config_opencat: dict, save: bool) -> pd.DataFrame:
  """
  """
  
  # load periodic table
  df_periodic_table = pd.read_csv(
    config_opencat['path_to_data_raw_periodic_table'])
  
  # set AtomicNumber column as index
  df_periodic_table = df_periodic_table.set_index(['AtomicNumber'])
  
  # transpose the dataframe
  df_periodic_table = df_periodic_table.transpose()
  
  # create type separated dataframes
  df_num_features = df_periodic_table.loc[config_opencat['num_features']]
  df_ord_features = df_periodic_table.loc[config_opencat['ord_features']]
  df_onehot_features = df_periodic_table.loc[config_opencat['onehot_features']]


  ### Process num_features ####
  # replace NaN entries in numeric features with zero
  df_num_features = df_num_features.fillna(0)
  
  
  ### Proecess ord_features ###
  # ordinally encode categorical features
  dict_encoding = {}
  for index in df_ord_features.index:
    codes, uniques = pd.factorize(df_ord_features.loc[index])
    df_ord_features.loc[index] = codes
    dict_encoding[index] = list(uniques)
  
  
  ### Process onehot_features ###
  
  # replace not available with 'unknown' category
  df_onehot_features = df_onehot_features.fillna('unknown')
  
  # create list of oxidation state entries
  list_ox_states = list(df_onehot_features.loc['OxidationStates'])

  # process list and create unique entries list
  dict_ox_states = {}
  new_list_ox_states = []
  for index, string in enumerate(list_ox_states):
      list_entries = []
      for entry in string.split(", "):
          # second split necessary because of irregularity in one case
          for entry_2 in entry.split(","):  
              new_list_ox_states.append(entry_2)
              list_entries.append(entry_2)
      
      dict_ox_states[index+1] = list_entries
      
  # create set and new index list
  set_ox_states = set(new_list_ox_states)
  new_index_list = []
  for ox_state in set_ox_states:
      new_index_list.append(ox_state)
  new_index_list.sort()

  # create new dataframe with one hot encodings
  df_onehot_features_new = pd.DataFrame(0, index=new_index_list, 
    columns=df_onehot_features.columns)
  for col, value in dict_ox_states.items():
      for index in value:
          df_onehot_features_new.loc[index, col] = 1

  # overwrite with new dataframe          
  df_onehot_features = df_onehot_features_new
  
  # save
  if save:
  
    # set saving paths
    p_table = config_opencat['path_to_data_metadata'] + 'periodic_table.csv'
    p_num = config_opencat['path_to_data_metadata'] + 'numeric_feat.csv'
    p_ord = config_opencat['path_to_data_metadata'] + 'ordinal_feat.csv'
    p_onehot = config_opencat['path_to_data_metadata'] + 'onehot_ox_feat.csv'
    p_ord_enc = config_opencat['path_to_data_metadata'] + 'ordinal_enc.json'
    
    # save files
    df_periodic_table.to_csv(p_table)
    df_num_features.to_csv(p_num)
    df_ord_features.to_csv(p_ord)
    df_onehot_features.to_csv(p_onehot)
    with open(p_ord_enc, 'w') as json_f:
      json.dump(dict_encoding, json_f)
    
  return df_periodic_table, df_num_features, df_ord_features, df_onehot_features
      
      
      
def load_dataset(path:str, single_lmdb:bool=False) -> (
  ocpmodels.datasets.lmdb_dataset.SinglePointLmdbDataset | 
  ocpmodels.datasets.lmdb_dataset.TrajectoryLmdbDataset):
  """
  """
  
  if single_lmdb:
    filename = os.listdir(path)[1]
    path += filename
    dataset = SinglePointLmdbDataset({"src": path})
    
  else:
    dataset = TrajectoryLmdbDataset({"src": path})
      
  return dataset    
  
  
  
def create_is2res_data(config_opencat: dict, path_list: list[str], save: bool,
  single_lmdb=False):
  """
  """

  # create empty dictionary to save results and initialize minimums and maximums
  is2res_train_dict = {}
  is2res_val_dict = {}
  is2res_test_dict = {}
  train_count, val_count, test_count = 0, 0, 0
  id_counter = 0
  max_atoms = 0
  min_atoms = 1e10
  
  # iterate over passed path list
  for path in path_list:
  
    # load dataset from currently iterated path
    is2res_dataset = load_dataset(path, single_lmdb)
    
    # subsample data
    n_data = len(is2res_dataset)
    n_samples = int(config_opencat['subsample_frac'] * n_data)
    random.seed(config_opencat['seed'])
    sample_list = random.sample(range(n_data), n_samples)
    
    # create array to calc min and max values
    atoms_array = np.zeros((len(sample_list),))
    entry_counter = 0
    
    # iterate over all datapoints
    for data_index in tqdm(sample_list):
      
      # get random datapoint
      datapoint = is2res_dataset[data_index]
      
      # save datapoint information in results dictionary
      if (config_opencat['spatial_ood']['n_atom_bounds'][0] < datapoint.natoms
        and datapoint.natoms < config_opencat['spatial_ood']['n_atom_bounds'][1]
      ):
        is2res_train_dict[id_counter] = {
          'relexed_energy' : datapoint.y_relaxed,
          'atomic_numbers' : datapoint.atomic_numbers.int().tolist(),
          'initial_structure' : datapoint.pos.tolist(),
          'relaxed_strucutre' : datapoint.pos_relaxed.tolist()
        }
        
      else:
        is2res_test_dict[id_counter] = {
          'relexed_energy' : datapoint.y_relaxed,
          'atomic_numbers' : datapoint.atomic_numbers.int().tolist(),
          'initial_structure' : datapoint.pos.tolist(),
          'relaxed_strucutre' : datapoint.pos_relaxed.tolist()
        }
      id_counter += 1
      
      # save number of atoms in system
      atoms_array[entry_counter] = datapoint.natoms
      entry_counter += 1
      
      
      ### every couple iterations, save some files and reduce length ###
      if entry_counter % (100 * config_opencat['data_per_file']) == 0:
        # shuffle dictionary entries
        is2res_test_dict =  list(is2res_test_dict.items())
        random.seed(config_opencat['seed'])
        random.shuffle(is2res_test_dict)
        is2res_test_dict = dict(is2res_test_dict)
        # split off validation data
        is2res_val_dict_add = dict(
          list(
            is2res_test_dict.items()
          )[:int(len(is2res_test_dict) * config_opencat['val_test_split'])]
        )
        # add to validation data
        is2res_val_dict = {**is2res_val_dict, **is2res_val_dict_add}
        # free up memory
        del is2res_val_dict_add
        gc.collect()
        # reset remaining as testing data
        is2res_test_dict = dict(
          list(
            is2res_test_dict.items()
          )[int(len(is2res_test_dict) * config_opencat['val_test_split']):]
        )
        
        ### Save
        is2res_train_dict, train_count = save_chunk(config_opencat, is2res_train_dict,
          train_count, config_opencat['path_to_data_subtask_train'], 'training', 
          save=save)
        is2res_val_dict, val_count = save_chunk(config_opencat, is2res_val_dict, 
          val_count, config_opencat['path_to_data_subtask_val'], 'validation', 
          save=save)
        is2res_test_dict, test_count = save_chunk(config_opencat, is2res_test_dict, 
          test_count, config_opencat['path_to_data_subtask_test'], 'testing', 
          save=save)
    
    # calculate min and max number of atoms
    min_atoms_array = atoms_array.min()
    max_atoms_array = atoms_array.max()
      
    # update min and max
    if min_atoms_array < min_atoms:
      min_atoms = int(min_atoms_array)
    if max_atoms_array > max_atoms:
      max_atoms = int(max_atoms_array)
      
      
    ### split off val here every couple of iteration
    # shuffle dictionary entries
    is2res_test_dict =  list(is2res_test_dict.items())
    random.seed(config_opencat['seed'])
    random.shuffle(is2res_test_dict)
    is2res_test_dict = dict(is2res_test_dict)
    # split off validation data
    is2res_val_dict_add = dict(
      list(
        is2res_test_dict.items()
      )[:int(len(is2res_test_dict) * config_opencat['val_test_split'])]
    )
    # add to validation data
    is2res_val_dict = {**is2res_val_dict, **is2res_val_dict_add}
    # free up memory
    del is2res_val_dict_add
    gc.collect()
    # reset remaining as testing data
    is2res_test_dict = dict(
      list(
        is2res_test_dict.items()
      )[int(len(is2res_test_dict) * config_opencat['val_test_split']):]
    )
    
    ### Save
    is2res_train_dict, train_count = save_chunk(config_opencat, 
      is2res_train_dict, train_count, 
      config_opencat['path_to_data_subtask_train'], 'training', save=save)
    is2res_val_dict, val_count = save_chunk(config_opencat, is2res_val_dict, 
      val_count, config_opencat['path_to_data_subtask_val'], 'validation', 
      save=save)
    is2res_test_dict, test_count = save_chunk(config_opencat, is2res_test_dict, 
      test_count, config_opencat['path_to_data_subtask_test'], 'testing', 
      save=save)
      
      
  # free up memory
  del is2res_dataset
  gc.collect()
  
  ### split off val here every couple of iteration
  # shuffle dictionary entries
  is2res_test_dict =  list(is2res_test_dict.items())
  random.seed(config_opencat['seed'])
  random.shuffle(is2res_test_dict)
  is2res_test_dict = dict(is2res_test_dict)
  # split off validation data
  is2res_val_dict_add = dict(
    list(
      is2res_test_dict.items()
    )[:int(len(is2res_test_dict) * config_opencat['val_test_split'])]
  )
  # add to validation data
  is2res_val_dict = {**is2res_val_dict, **is2res_val_dict_add}
  # free up memory
  del is2res_val_dict_add
  gc.collect()
  # reset remaining as testing data
  is2res_test_dict = dict(
    list(
      is2res_test_dict.items()
    )[int(len(is2res_test_dict) * config_opencat['val_test_split']):]
  )
    
  ### Calculate n_train, n_val, n_test here
  n_train = len(is2res_train_dict) + train_count * config_opencat['data_per_file']
  n_val = len(is2res_val_dict) + val_count * config_opencat['data_per_file']
  n_test = len(is2res_test_dict) + test_count * config_opencat['data_per_file']
   
   
  ### Save
  is2res_train_dict, train_count = save_chunk(config_opencat, is2res_train_dict,
    train_count, config_opencat['path_to_data_subtask_train'], 'training', 
    save=save, last_iteration=True)
  is2res_val_dict, val_count = save_chunk(config_opencat, is2res_val_dict, 
    val_count, config_opencat['path_to_data_subtask_val'], 'validation', 
    save=save, last_iteration=True)
  is2res_test_dict, test_count = save_chunk(config_opencat, is2res_test_dict, 
    test_count, config_opencat['path_to_data_subtask_test'], 'testing', 
    save=save, last_iteration=True)
  
  
  # print informative values
  print("\nNumber of atoms are: {} - {}\n".format(min_atoms, max_atoms))
  print_split_results(n_train, n_val, n_test)
  



def create_s2ef_data(config_opencat: dict, path_list: list[str], save: bool):
  """
  """
  
  # create empty dictionary to save results
  s2ef_train_dict = {}
  s2ef_val_dict = {}
  s2ef_test_dict = {}
  train_count, val_count, test_count = 0, 0, 0
  id_counter = 0
  max_atoms = 0
  min_atoms = 1e10
  
  # iterate over passed path list
  for path in path_list:
  
    # load dataset from currently iterated path
    s2ef_dataset = load_dataset(path)
    
    # subsample data
    n_data = len(s2ef_dataset)
    n_samples = int(config_opencat['subsample_frac'] * n_data)
    random.seed(config_opencat['seed'])
    sample_list = random.sample(range(n_data), n_samples)
    
    # create array to calc min and max values
    atoms_array = np.zeros((len(sample_list),))
    entry_counter = 0
    
    # iterate over all datapoints
    for data_index in tqdm(sample_list):
      
      # get random datapoint
      datapoint = s2ef_dataset[data_index]
      
      # save datapoint information in results dictionary
      if (config_opencat['spatial_ood']['n_atom_bounds'][0] < datapoint.natoms
        and datapoint.natoms < config_opencat['spatial_ood']['n_atom_bounds'][1]
      ):
        s2ef_train_dict[id_counter] = {
          'energy' : datapoint.y,
          'atomic_numbers' : datapoint.atomic_numbers.int().tolist(),
          'structure' : datapoint.pos.tolist(),
          'forces' : datapoint.force.tolist()
        }
        
      else:
        s2ef_test_dict[id_counter] = {
          'energy' : datapoint.y,
          'atomic_numbers' : datapoint.atomic_numbers.int().tolist(),
          'structure' : datapoint.pos.tolist(),
          'forces' : datapoint.force.tolist()
        }
      id_counter += 1
      
      # save number of atoms in system
      atoms_array[entry_counter] = datapoint.natoms
      entry_counter += 1
      
      
      ### every couple iterations, save some files and reduce length ###
      if entry_counter % (100 * config_opencat['data_per_file']) == 0:
        # shuffle dictionary entries
        s2ef_test_dict =  list(s2ef_test_dict.items())
        random.seed(config_opencat['seed'])
        random.shuffle(s2ef_test_dict)
        s2ef_test_dict = dict(s2ef_test_dict)
        # split off validation data
        s2ef_val_dict_add = dict(
          list(
            s2ef_test_dict.items()
          )[:int(len(s2ef_test_dict) * config_opencat['val_test_split'])]
        )
        # add to validation data
        s2ef_val_dict = {**s2ef_val_dict, **s2ef_val_dict_add}
        # free up memory
        del s2ef_val_dict_add
        gc.collect()
        # reset remaining as testing data
        s2ef_test_dict = dict(
          list(
            s2ef_test_dict.items()
          )[int(len(s2ef_test_dict) * config_opencat['val_test_split']):]
        )
        
        ### Save
        s2ef_train_dict, train_count = save_chunk(config_opencat, s2ef_train_dict,
          train_count, config_opencat['path_to_data_subtask_train'], 'training', 
          save=save)
        s2ef_val_dict, val_count = save_chunk(config_opencat, s2ef_val_dict, 
          val_count, config_opencat['path_to_data_subtask_val'], 'validation', 
          save=save)
        s2ef_test_dict, test_count = save_chunk(config_opencat, s2ef_test_dict, 
          test_count, config_opencat['path_to_data_subtask_test'], 'testing', 
          save=save)
    
    # calculate min and max number of atoms
    min_atoms_array = atoms_array.min()
    max_atoms_array = atoms_array.max()
      
    # update min and max
    if min_atoms_array < min_atoms:
      min_atoms = int(min_atoms_array)
    if max_atoms_array > max_atoms:
      max_atoms = int(max_atoms_array)
      
      
    ### split off val here every couple of iteration
    # shuffle dictionary entries
    s2ef_test_dict =  list(s2ef_test_dict.items())
    random.seed(config_opencat['seed'])
    random.shuffle(s2ef_test_dict)
    s2ef_test_dict = dict(s2ef_test_dict)
    # split off validation data
    s2ef_val_dict_add = dict(
      list(
        s2ef_test_dict.items()
      )[:int(len(s2ef_test_dict) * config_opencat['val_test_split'])]
    )
    # add to validation data
    s2ef_val_dict = {**s2ef_val_dict, **s2ef_val_dict_add}
    # free up memory
    del s2ef_val_dict_add
    gc.collect()
    # reset remaining as testing data
    s2ef_test_dict = dict(
      list(
        s2ef_test_dict.items()
      )[int(len(s2ef_test_dict) * config_opencat['val_test_split']):]
    )
    
    ### Save
    s2ef_train_dict, train_count = save_chunk(config_opencat, s2ef_train_dict,
      train_count, config_opencat['path_to_data_subtask_train'], 'training', 
      save=save)
    s2ef_val_dict, val_count = save_chunk(config_opencat, s2ef_val_dict, 
      val_count, config_opencat['path_to_data_subtask_val'], 'validation', 
      save=save)
    s2ef_test_dict, test_count = save_chunk(config_opencat, s2ef_test_dict, 
      test_count, config_opencat['path_to_data_subtask_test'], 'testing', 
      save=save)
      
      
  # free up memory
  del s2ef_dataset
  gc.collect()
  
  ### split off val here every couple of iteration
  # shuffle dictionary entries
  s2ef_test_dict =  list(s2ef_test_dict.items())
  random.seed(config_opencat['seed'])
  random.shuffle(s2ef_test_dict)
  s2ef_test_dict = dict(s2ef_test_dict)
  # split off validation data
  s2ef_val_dict_add = dict(
    list(
      s2ef_test_dict.items()
    )[:int(len(s2ef_test_dict) * config_opencat['val_test_split'])]
  )
  # add to validation data
  s2ef_val_dict = {**s2ef_val_dict, **s2ef_val_dict_add}
  # free up memory
  del s2ef_val_dict_add
  gc.collect()
  # reset remaining as testing data
  s2ef_test_dict = dict(
    list(
      s2ef_test_dict.items()
    )[int(len(s2ef_test_dict) * config_opencat['val_test_split']):]
  )
    
  ### Calculate n_train, n_val, n_test here
  n_train = len(s2ef_train_dict) + train_count * config_opencat['data_per_file']
  n_val = len(s2ef_val_dict) + val_count * config_opencat['data_per_file']
  n_test = len(s2ef_test_dict) + test_count * config_opencat['data_per_file']
   
   
  ### Save
  s2ef_train_dict, train_count = save_chunk(config_opencat, s2ef_train_dict,
    train_count, config_opencat['path_to_data_subtask_train'], 'training', 
    save=save, last_iteration=True)
  s2ef_val_dict, val_count = save_chunk(config_opencat, s2ef_val_dict, 
    val_count, config_opencat['path_to_data_subtask_val'], 'validation', 
    save=save, last_iteration=True)
  s2ef_test_dict, test_count = save_chunk(config_opencat, s2ef_test_dict, 
    test_count, config_opencat['path_to_data_subtask_test'], 'testing', 
    save=save, last_iteration=True)
  
  
  # print informative values
  print("\nNumber of atoms are: {} - {}\n".format(min_atoms, max_atoms))
  print_split_results(n_train, n_val, n_test)

  

def save_chunk(config_opencat: dict, data_dict: dict, chunk_counter: int,
  saving_path: str, filename: str, save: bool, last_iteration=False):
  """
  Save a chunk of dictionary in case it exceeds the number of data points per
  file that we want, or in case we are in the last iteration.
  """
  
  ### Save resulting data in chunks
  while len(data_dict) > config_opencat['data_per_file'] or last_iteration:
    # increment chunk counter 
    chunk_counter += 1
    
    # create path
    path_to_saving = '{}_{}.json'.format(saving_path + filename, chunk_counter)
    
    # shuffle
    data_dict =  list(data_dict.items())
    random.seed(config_opencat['seed'])
    random.shuffle(data_dict)
    data_dict = dict(data_dict)
    
    # save chunk
    if save and len(data_dict) > 0:
      
      # set saving dict
      save_dict = dict(
        list(
          data_dict.items()
        )[:config_opencat['data_per_file']]
      )
      
      # save to json file
      with open(path_to_saving, 'w') as w_file:
        json.dump(save_dict, w_file)
      
    # delete saved chunk
    data_dict = dict(
      list(
        data_dict.items()
      )[config_opencat['data_per_file']:]
    )
    
    # must be set to exit loop on last iteration
    last_iteration = False
      
  return data_dict, chunk_counter 
  
  
def print_split_results(n_train: int, n_val: int, n_test: int):
  """
  """
  
  # set total number
  n_total = n_train + n_val + n_test
  
  # print
  print("Training data   :   {}/{} {:.0%}".format(n_train, n_total, 
      n_train/n_total),
    "\nValidation data :   {}/{} {:.0%}".format(n_val, n_total,
      n_val/n_total),
    "\nTesting data    :   {}/{} {:.0%}".format(n_test, n_total,
      n_test/n_total)
  )
