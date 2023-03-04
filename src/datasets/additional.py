import os
import pandas as pd



def shuffle_data_files(config_dataset: dict, n_iter_shuffle=3, 
    n_files_simultan=10):
    """
    shuffles data for a passed dataset configuration. Assumes that data is 
    available on the standard paths.
    """
    
    # set the list of folder paths you want to shuffle. 
    path_to_folder_list = [
        config_dataset['path_to_data_subtask_train'],
        config_dataset['path_to_data_subtask_val'],
        config_dataset['path_to_data_subtask_test']]
    
    # create progress bar
    pbar = tqdm(total=n_iter_shuffle*len(path_to_folder_list))
    
    # do this for train, val and test datasets separately
    for path_to_folder in path_to_folder_list:
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
            random.seed(config_dataset['seed'])
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
            df = df.sample(frac=1, random_state=config_dataset['seed'])
            
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


















    


    
