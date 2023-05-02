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
        df, n_data_points_list = load_csv_fast(path_to_folder, sampled_files)
        
        # shuffle
        df = df.sample(frac=1, random_state=config['general']['seed'])
        
                
        # write csv fast
        write_csv_fast(df, sampled_files, n_data_points_list, path_to_folder)
        
        
"""        
        # declare empty dataframe
        df = pd.DataFrame()
        
        # declare empty list to save number of data points of each file
        n_data_points_list = []
        
        
        # iterate over all sampled files
        for filename in sampled_files:
        
          # create path to iterated file
          path_to_csv = path_to_folder + filename
          
          # import iterated file
          df_csv = pd.read_csv(path_to_csv)
          
          # track the number of data points available in file
          n_data_points_list.append(len(df_csv.index))
          
          # append imported file to dataframe
          df = pd.concat([df, df_csv])
        
        # free up memory
        del df_csv
        gc.collect()
        
        # shuffle
        df = df.sample(frac=1, random_state=config['general']['seed'])
        
        # iterate over sampled files and n_data_points simultaneously
        for filename, n_data_points in zip(sampled_files, n_data_points_list):
        
          # create path to iterated file
          path_to_csv = path_to_folder + filename
          
          # save shuffled slice
          df[:n_data_points].to_csv(path_to_csv, index=False)
          
          # remove saved slice
          df = df[n_data_points:]
            
        # update progress bar
        pbar.update(1) 
"""
        

def write_csv_fast(df: pd.DataFrame, sampled_files: list[str], 
  n_data_points_list: list[int], path_to_folder: str):
  """
  """

  # define function to parallelize
  def write_csv(path_to_csv, df_slice, start, end):
    
    # save df slice
    df_slice.to_csv(path_to_csv, index=False)
  
  
  # open parall execution thread pool
  with ThreadPoolExecutor() as executor:
    
    # iterate over lists and add to execution pool
    for fname, n_samples in zip(sampled_files, n_data_points_list):
      
      # set full saving path argument 1
      path_to_csv = path_to_folder + fname
      
      # execute
      executor.submit(write_csv, df[:n_samples], path_to_csv, start, end)
      
      # shorten df
      df = df[n_samples:]
      
      
  
def load_csv_fast(path_to_folder: str, filenames: list[str]) -> pd.DataFrame:
  """
  """
  
  # define function to parallelize
  def load_csv(path_to_csv):
    
    # read csv
    df = pd.read_csv(path_to_csv)
    
    # remove csv. This makes writing faster
    os.remove(path_to_csv)
    
    return df

  # open parall execution thread pool
  with ThreadPoolExecutor() as executor:
    
    futures = []
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











    


    
