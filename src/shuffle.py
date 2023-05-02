import os
import pandas as pd
from tqdm import tqdm
import random
import gc
from concurrent.futures import ThreadPoolExecutor


def shuffle_data_files(name: str, config: dict, n_iter_shuffle=1, 
  n_files_simultan=600):
  """
  shuffles data for a passed dataset configuration. Assumes that data is 
  available on the standard paths.
  """
  print("Shuffling processed {} data.".format(name))
  
  # iterate over all subtasks
  for subtask in ['cities_43']: #config[name]['subtask_list']:
    
    # set some paths
    path_to_train = (config['general']['path_to_data'] 
      + name + '/' + subtask + '/training/')
    path_to_val = (config['general']['path_to_data'] 
      + name + '/' + subtask + '/validation/')
    path_to_test = (config['general']['path_to_data'] 
      + name + '/' + subtask + '/testing/')
    
    
    # do this for train, val and test datasets separately
    for path_to_folder in [path_to_test]: #[path_to_train, path_to_val, path_to_test]:
    
      # get a list of files in currently iterated dataset (train,val, or test)
      file_list = os.listdir(path_to_folder)
      
      # determine number of samples in dependence on file list length
      if n_files_simultan > len(file_list):
        n_samples = len(file_list)
        
      else:
        n_samples = n_files_simultan
          
      # do this for n_iter_shuffle times
      for _ in range(n_iter_shuffle):
      
        # randomly sample n_samples from file list
        random.seed(config['general']['seed'])
        sampled_files = random.sample(file_list, n_samples)
        
        # load csv fast
        print("\nLoading .csv files now!")
        df, n_data_points_list = load_csv_fast(path_to_folder, sampled_files)
        
        # shuffle
        print("\nShuffling dataframe now!")
        df = df.sample(frac=1, random_state=config['general']['seed'])
        
        # write csv fast
        print("\nWriting dataframe to .csv again now!")
        write_csv_fast(path_to_folder, df, sampled_files, n_data_points_list)
        
        
        

def write_csv_fast(path_to_folder: str, df: pd.DataFrame, 
  sampled_files: list[str], n_data_points_list: list[int]):
  """
  """

  # define function to parallelize
  def write_csv(path_to_csv, df_slice):
    
    # save df slice
    df_slice.to_csv(path_to_csv, index=False)
  
  # open parall execution thread pool
  with ThreadPoolExecutor() as executor:
    
    # iterate over lists and add to execution pool
    for fname, n_samples in zip(sampled_files, n_data_points_list):
      
      # set full saving path argument 1
      path_to_csv = path_to_folder + fname
      
      # execute
      executor.submit(write_csv, path_to_csv, df[:n_samples])
      
      # shorten df
      df = df[n_samples:]
      
      
      
def load_csv_fast(path_to_folder: str, filenames: list[str]) -> pd.DataFrame:
  """
  """
  
  # define function to parallelize
  def load_csv(path_to_csv):
    
    return pd.read_csv(path_to_csv)

  # open parall execution thread pool
  with ThreadPoolExecutor() as executor:
    
    # declare list to save results
    futures = []
    
    # iterate over lists and add to execution pool
    for fname in filenames:
      
      # add to executor and save future results in list
      path_to_csv = path_to_folder + fname
      futures.append(executor.submit(load_csv, path_to_csv))
    
    # create empty lists to read results
    dfs = []
    n_data_points_list = []
    
    # iterate over all parallelzed execution results
    for f in tqdm(futures):
      
      # create lists from results
      n_data_points_list.append(len(f.result().index))
      dfs.append(f.result())
  
  # concatenate dataframes
  df_result = pd.concat(dfs, ignore_index=True, copy=False)
  
  return df_result, n_data_points_list











    


    
