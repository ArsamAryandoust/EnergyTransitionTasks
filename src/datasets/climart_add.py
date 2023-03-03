import os
import numpy as np
import json
import h5py



def visualize_raw_keys_and_shapes(dataset, name):
    
    """ """
    dataset_key_list = list(dataset.keys())
    print('\nKeys of {} data:\n{}'.format(name, dataset_key_list))
    
    for key in dataset_key_list:
        value = dataset[key]
        print('\n{}: \n{}'.format(key, value.shape))



def import_data_stats(HYPER):

    """ """
    
    # get list of all stats files
    stats_filename_list = os.listdir(HYPER.PATH_TO_DATA_RAW_CLIMART_STATISTICS)
    
    # declare empty stats dictionary to store all files
    stats_dict = {}
    
    # iterate over all file names
    for filename in stats_filename_list:
        
        # set the full path to file
        path_to_file = HYPER.PATH_TO_DATA_RAW_CLIMART_STATISTICS + filename
        
        # variable name
        variable_name = filename[:-4]
        
        # import data
        stats_dict[variable_name] = np.load(path_to_file)
    
    return stats_dict
    
    


    
def shuffle_data_files(
    HYPER,
    n_iter_shuffle=3,
    n_files_simultan=10
):

    """ """
    path_to_folder_list = [
        HYPER.PATH_TO_DATA_CLIMART_CLEARSKY_TRAIN,
        HYPER.PATH_TO_DATA_CLIMART_PRISTINE_TRAIN,
        HYPER.PATH_TO_DATA_CLIMART_CLEARSKY_VAL,
        HYPER.PATH_TO_DATA_CLIMART_PRISTINE_VAL,
        HYPER.PATH_TO_DATA_CLIMART_CLEARSKY_TEST,
        HYPER.PATH_TO_DATA_CLIMART_PRISTINE_TEST
    ]
    
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
            random.seed(HYPER.SEED)
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
            
            # empty storage
            del df_csv
            gc.collect()
            
            # shuffle
            df = df.sample(frac=1, random_state=HYPER.SEED)
            
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
