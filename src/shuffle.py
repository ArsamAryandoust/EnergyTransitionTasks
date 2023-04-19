import os
import pandas as pd
from tqdm import tqdm
import random
import gc


def shuffle_data_files(name: str, config: dict, n_iter_shuffle=1, 
  n_files_simultan=200):
  """
  shuffles data for a passed dataset configuration. Assumes that data is 
  available on the standard paths.
  """
  print("Shuffling processed {} data.".format(name))
  
  # iterate over all subtasks
  for subtask in config[name]['subtask_list']:
    
    # set some paths
    path_to_train = (config['general']['path_to_data'] 
      + name + '/' + subtask + '/training/')
    path_to_val = (config['general']['path_to_data'] 
      + name + '/' + subtask + '/validation/')
    path_to_test = (config['general']['path_to_data'] 
      + name + '/' + subtask + '/testing/')
    
    # create progress bar
    pbar = tqdm(total=n_iter_shuffle*3)
    
    # do this for train, val and test datasets separately
    for path_to_folder in [path_to_train, path_to_val, path_to_test]:
    
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

















    


    
