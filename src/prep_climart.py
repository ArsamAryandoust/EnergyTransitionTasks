import os
import numpy as np
import json
import h5py
import pandas as pd
import gc
import math
import random
from tqdm import tqdm


def create_input_col_name_list(
    var_dict
):
    
    """ """

    # declare empty column names list
    col_names_list = []
    
    # iterate over variable name dict
    for var_name in var_dict:
        
        # get dict with start and end range of current variable
        var_range_dict = var_dict[var_name]
        
        # get range size
        range_size = var_range_dict['end'] - var_range_dict['start']
        
        # append column name to list if range size is only 1
        if range_size == 1:
            col_names_list.append(var_name)
            
        elif range_size >1:
            
            for feature_iter in range(range_size):
                col_name = var_name + '_{}'.format(feature_iter)
                col_names_list.append(col_name)
                
        else:
        
            print('Caution. Something went wrong with creating column names')
    
    return col_names_list



def create_output_col_name_list(
    var_name,
    var_dict
):
    
    """ """
    
    # get variable range dictionary
    var_range_dict = var_dict[var_name]
    
    # declare empty column names list
    col_names_list = []
    
    # get range size
    range_size = var_range_dict['end'] - var_range_dict['start']
    
    # append column name to list if range size is only 1
    for feature_iter in range(range_size):
        col_name = var_name + '_{}'.format(feature_iter)
        col_names_list.append(col_name)

    return col_names_list


def process_raw_data(
    feature_by_var,
    inputs, 
    outputs_clear_sky, 
    outputs_pristine
):
    
    """ """
    
    ###
    # Process input data ###
    ###
    
    # define empty dataframes
    df_inputs_clear_sky = pd.DataFrame()
    df_inputs_pristine = pd.DataFrame()
    
    ### Do for globals ###
    
    # retrieve data and tranform into numpy arrays
    data = np.array(inputs['globals'])
    
    # create column names
    var_dict = feature_by_var['globals']
    col_names_list = create_input_col_name_list(var_dict)
    
    # transform into dataframe
    data = pd.DataFrame(data, columns=col_names_list)
    
    # append to input dataframes
    df_inputs_clear_sky = pd.concat([df_inputs_clear_sky, data], axis=1)
    df_inputs_pristine = pd.concat([df_inputs_pristine, data], axis=1)
    
    
    ### Do for layers ###
    
    # retrieve data and tranform into numpy arrays
    data = np.array(inputs['layers'])
    data_pristine = np.array(inputs['layers'][:, :, :14])
    
    # create column names
    var_dict = feature_by_var['layers']
    col_names_list = create_input_col_name_list(var_dict)
    col_names_list_pristine = col_names_list[:14].copy()
    
    # reshape data
    data, col_names_list = reshape(data, col_names_list)
    data_pristine, col_names_list_pristine = reshape(data_pristine, col_names_list_pristine)
    
    # transform into dataframe
    data = pd.DataFrame(data, columns=col_names_list)
    data_pristine = pd.DataFrame(data_pristine, columns=col_names_list_pristine)
    
    # append to input dataframes
    df_inputs_clear_sky = pd.concat([df_inputs_clear_sky, data], axis=1) 
    df_inputs_pristine = pd.concat([df_inputs_pristine, data_pristine], axis=1) 
    
    # free up memory
    del data_pristine
    gc.collect()
    
    
    ### Do for levels ###
    
    # retrieve data and tranform into numpy arrays
    data = np.array(inputs['levels'])
    
    # create column names
    var_dict = feature_by_var['levels']
    col_names_list = create_input_col_name_list(var_dict)
    
    # reshape data
    data, col_names_list = reshape(data, col_names_list)
    
    # transform into dataframe
    data = pd.DataFrame(data, columns=col_names_list)
    
    # append to input dataframes
    df_inputs_clear_sky = pd.concat([df_inputs_clear_sky, data], axis=1)
    df_inputs_pristine = pd.concat([df_inputs_pristine, data], axis=1)
    
    # free up memory
    del inputs
    gc.collect()
    
    
    ###
    # Process output data ###
    ###
    
    # define empty dataframes
    df_outputs_clear_sky = pd.DataFrame()
    df_outputs_pristine = pd.DataFrame()
    
    # iterate over both outputs simultaneously
    for key_clear_sky, key_pristine in zip(outputs_clear_sky, outputs_pristine):
        
        # retrieve data and tranform into numpy arrays
        data_clear_sky = np.array(outputs_clear_sky[key_clear_sky])
        data_pristine = np.array(outputs_pristine[key_pristine])
        
        # create column names
        var_dict_clear_sky = feature_by_var['outputs_clear_sky']
        col_names_list_outputs_clear_sky = create_output_col_name_list(key_clear_sky, var_dict_clear_sky)
        
        var_dict_pristine = feature_by_var['outputs_pristine']
        col_names_list_output_pristine = create_output_col_name_list(key_pristine, var_dict_pristine)
        
        # transform into dataframe
        data_clear_sky = pd.DataFrame(data_clear_sky, columns=col_names_list_outputs_clear_sky)
        data_pristine = pd.DataFrame(data_pristine, columns=col_names_list_output_pristine)
        
        # append to input dataframes
        df_outputs_clear_sky = pd.concat([df_outputs_clear_sky, data_clear_sky], axis=1)
        df_outputs_pristine = pd.concat([df_outputs_pristine, data_pristine], axis=1)
    
    
    return df_inputs_clear_sky, df_inputs_pristine, df_outputs_clear_sky, df_outputs_pristine



def reshape(data, col_names_list):

    """ """
    
    # get number of data points
    n_data = len(data)
    n_steps = data.shape[1]
    n_features = data.shape[2]
    
    # get number of columns for desired reshaping
    n_columns = n_steps * n_features
    
    # reshape with C order
    data = np.reshape(data, (n_data, n_columns), order='C')
    
    # declare new empty column names list
    expanded_col_names_list = []
    
    # expand col_names_list according to reshape with C order
    for steps in range(n_steps):
        
        # iterate over all column names
        for col_name in col_names_list:
            
            # create entry
            entry= 'l_{}_{}'.format(steps, col_name)
            
            # append entry
            expanded_col_names_list.append(entry)
            
        
    return data, expanded_col_names_list



def import_h5_data(HYPER, year):

    """ """
    
    # create paths to files
    path_to_inputs = (
        HYPER.PATH_TO_DATA_RAW_CLIMART_INPUTS
        + year
        + '.h5'
    )
    path_to_outputs_clear_sky = (
        HYPER.PATH_TO_DATA_RAW_CLIMART_OUTPUTS_CLEAR_SKY
        + year
        + '.h5'
    )
    path_to_outputs_pristine = (
        HYPER.PATH_TO_DATA_RAW_CLIMART_OUTPUTS_PRISTINE
        + year
        + '.h5'
    )
    
    # load data
    inputs = h5py.File(path_to_inputs, 'r')
    outputs_clear_sky = h5py.File(path_to_outputs_clear_sky, 'r')
    outputs_pristine = h5py.File(path_to_outputs_pristine, 'r')
    
    
    return inputs, outputs_clear_sky, outputs_pristine
    


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
    
    
def import_meta_json(HYPER):

    """ """
    
    # create path to json file
    path_to_meta_json = HYPER.PATH_TO_DATA_RAW_CLIMART + 'META_INFO.json'
    
    # load json file
    with open(path_to_meta_json, 'r') as f:
        meta_dict = json.load(f)
    
    
    input_dims = meta_dict['input_dims']
    variables = meta_dict['variables']
    feature_by_var = meta_dict['feature_by_var']
    
    return input_dims, variables, feature_by_var
    
    
def augment_and_merge(
    year,
    df_inputs_clear_sky, 
    df_inputs_pristine, 
    df_outputs_clear_sky, 
    df_outputs_pristine
):

    """ """
    
    # concatenate dataframes
    df_clear_sky = pd.concat([df_inputs_clear_sky, df_outputs_clear_sky], axis=1)
    df_pristine = pd.concat([df_inputs_pristine, df_outputs_pristine], axis=1)
    
    # calculate for each data point the hour of year
    n_lat, n_lon, n_hours_per_step = 64, 128, 205
    n_points_space = n_lat * n_lon
    hour_of_year = (
        np.floor(df_clear_sky.index.values / n_points_space) * n_hours_per_step
    ).astype(int)
    
    # augment data with year
    df_clear_sky.insert(0, 'year', int(year))
    df_pristine.insert(0, 'year', int(year))
    
    # augment data with hour of year
    df_clear_sky.insert(1, 'hour_of_year', hour_of_year)
    df_pristine.insert(1, 'hour_of_year', hour_of_year)
    
    
    ### put re-arange order of spatial coordinates for visibility ###
    
    # get column name lists
    cols_clear_sky = df_clear_sky.columns.tolist()
    cols_pristine = df_pristine.columns.tolist()
    
    # remove entries from column list
    cols_clear_sky.remove('x_cord')
    cols_clear_sky.remove('y_cord')
    cols_clear_sky.remove('z_cord')
    
    cols_pristine.remove('x_cord')
    cols_pristine.remove('y_cord')
    cols_pristine.remove('z_cord')
    
    # insert them again at new spots    
    cols_clear_sky.insert(2,'x_cord')
    cols_clear_sky.insert(3,'y_cord')
    cols_clear_sky.insert(4,'z_cord')
    
    cols_pristine.insert(2,'x_cord')
    cols_pristine.insert(3,'y_cord')
    cols_pristine.insert(4,'z_cord')
    
    # set the new column orders
    df_clear_sky = df_clear_sky[cols_clear_sky]
    df_pristine = df_pristine[cols_pristine]
    
    
    return df_clear_sky, df_pristine
    
    
def train_val_test_split(
    HYPER
):

    """ """
    
    # list all data available in one of the raw data foulders.
    list_of_years = os.listdir(HYPER.PATH_TO_DATA_RAW_CLIMART_INPUTS)
    
    # remove file ending from list of years and turn into integer values
    list_of_years = [list_entry[:-3] for list_entry in list_of_years]
    
    # import meta data
    _, _, feature_by_var = import_meta_json(HYPER)
    
    # decleare empty dataframes for trainining validation and testing
    df_train_clear_sky = pd.DataFrame()
    df_train_pristine = pd.DataFrame()
    
    df_val_clear_sky = pd.DataFrame()
    df_val_pristine = pd.DataFrame()
    
    df_test_clear_sky = pd.DataFrame()
    df_test_pristine = pd.DataFrame()
    
    # declare data point counters
    train_chunk_counter_clearsky, val_chunk_counter_clearsky, test_chunk_counter_clearsky = 0, 0, 0
    train_chunk_counter_pristine, val_chunk_counter_pristine, test_chunk_counter_pristine = 0, 0, 0
    
    for year in list_of_years:
    
        # tell us which year we are processing
        print('Processing data for year {}'.format(year))
        # import inputs and outputs
        inputs, outputs_clear_sky, outputs_pristine = import_h5_data(HYPER, year)
        
        # process raw data
        (
            df_inputs_clear_sky, 
            df_inputs_pristine, 
            df_outputs_clear_sky, 
            df_outputs_pristine
        ) = process_raw_data(
            feature_by_var,
            inputs, 
            outputs_clear_sky, 
            outputs_pristine
        )
        
        # free up memory
        inputs.close()
        outputs_clear_sky.close()
        outputs_pristine.close()
        del inputs, outputs_clear_sky, outputs_pristine
        gc.collect()
    
        # augment and merge data
        df_clear_sky, df_pristine = augment_and_merge(
            year,
            df_inputs_clear_sky, 
            df_inputs_pristine, 
            df_outputs_clear_sky, 
            df_outputs_pristine
        )
        
        # free up memory
        del df_inputs_clear_sky, df_inputs_pristine 
        del df_outputs_clear_sky, df_outputs_pristine
        gc.collect()
        
        # subsample data
        df_clear_sky = df_clear_sky.sample(
            frac=HYPER.SUBSAMPLE_CLIMART,
            random_state=HYPER.SEED
        )
        df_pristine = df_pristine.sample(
            frac=HYPER.SUBSAMPLE_CLIMART,
            random_state=HYPER.SEED
        )
        
        # check if this is a testing year data
        if HYPER.TEST_SPLIT_DICT_CLIMART['temporal_dict']['year']==int(year):
            
            # append entire datasets to test dataframes
            df_test_clear_sky = pd.concat([df_test_clear_sky, df_clear_sky])
            df_test_pristine = pd.concat([df_test_pristine, df_pristine])
            
            # free up memory
            del df_clear_sky, df_pristine 
            gc.collect()
             
        else:
        
            # extract the rows from dataframes with indices for test coordinates
            df_test_coordiantes_clear_sky = df_clear_sky[
                df_clear_sky.index.isin(
                    HYPER.TEST_SPLIT_DICT_CLIMART['spatial_dict']['coordinates']
                )
            ]
            df_test_coordiantes_pristine = df_pristine[
                df_pristine.index.isin(
                    HYPER.TEST_SPLIT_DICT_CLIMART['spatial_dict']['coordinates']
                )
            ]
            
            # append extracted rows to test dataframes
            df_test_clear_sky = pd.concat([df_test_clear_sky, df_test_coordiantes_clear_sky])
            df_test_pristine = pd.concat([df_test_pristine, df_test_coordiantes_pristine])
            
            # set the remaining rows for training and validation
            df_clear_sky = df_clear_sky.drop(df_test_coordiantes_clear_sky.index)
            df_pristine = df_pristine.drop(df_test_coordiantes_pristine.index)
            
            # free up memory
            del df_test_coordiantes_clear_sky, df_test_coordiantes_pristine
            gc.collect()
            
            # extract the rows from dataframes with matching hours of year
            df_test_hours_of_year_clear_sky = df_clear_sky.loc[
                df_clear_sky['hour_of_year'].isin(
                    HYPER.TEST_SPLIT_DICT_CLIMART['temporal_dict']['hours_of_year']
                )
            ]
            df_test_hours_of_year_pristine = df_pristine.loc[
                df_pristine['hour_of_year'].isin(
                    HYPER.TEST_SPLIT_DICT_CLIMART['temporal_dict']['hours_of_year']
                )
            ]
            
            # append extracted rows to test dataframes
            df_test_clear_sky = pd.concat([df_test_clear_sky, df_test_hours_of_year_clear_sky])
            df_test_pristine = pd.concat([df_test_pristine, df_test_hours_of_year_pristine])
        
            # set the remaining rows for training and validation
            df_clear_sky = df_clear_sky.drop(df_test_hours_of_year_clear_sky.index)
            df_pristine = df_pristine.drop(df_test_hours_of_year_pristine.index)
            
            # free up memory
            del df_test_hours_of_year_clear_sky, df_test_hours_of_year_pristine
            gc.collect()
            
            
            # create training and validation datasets
            df_train_append_clear_sky = df_clear_sky.sample(
                frac=HYPER.TRAIN_VAL_SPLIT_CLIMART,
                random_state=HYPER.SEED
            )
            df_train_append_pristine = df_pristine.sample(
                frac=HYPER.TRAIN_VAL_SPLIT_CLIMART,
                random_state=HYPER.SEED
            )
            df_val_append_clear_sky = df_clear_sky.drop(df_train_append_clear_sky.index)
            df_val_append_pristine = df_pristine.drop(df_train_append_pristine.index)
            
            # append training dataset
            df_train_clear_sky = pd.concat([df_train_clear_sky, df_train_append_clear_sky])
            df_train_pristine = pd.concat([df_train_pristine, df_train_append_pristine])
            
            # free up memory     
            del df_train_append_clear_sky, df_train_append_pristine
            gc.collect()
        
            # append validation dataset
            df_val_clear_sky = pd.concat([df_val_clear_sky, df_val_append_clear_sky])
            df_val_pristine = pd.concat([df_val_pristine, df_val_append_pristine])
            
            # free up memory     
            del df_val_append_clear_sky, df_val_append_pristine
            gc.collect()
            
            
        ### Save resulting data in chunks
        df_train_clear_sky, train_chunk_counter_clearsky = save_chunk(
            HYPER,
            df_train_clear_sky,
            train_chunk_counter_clearsky,
            HYPER.PATH_TO_DATA_CLIMART_CLEARSKY_TRAIN,
            'training'    
        )
        df_train_pristine, train_chunk_counter_pristine = save_chunk(
            HYPER,
            df_train_pristine,
            train_chunk_counter_pristine,
            HYPER.PATH_TO_DATA_CLIMART_PRISTINE_TRAIN,
            'training'    
        )
        df_val_clear_sky, val_chunk_counter_clearsky = save_chunk(
            HYPER,
            df_val_clear_sky,
            val_chunk_counter_clearsky,
            HYPER.PATH_TO_DATA_CLIMART_CLEARSKY_VAL,
            'validation'
        )
        df_val_pristine, val_chunk_counter_pristine = save_chunk(
            HYPER,
            df_val_pristine,
            val_chunk_counter_pristine,
            HYPER.PATH_TO_DATA_CLIMART_PRISTINE_VAL,
            'validation'
        )
        df_test_clear_sky, test_chunk_counter_clearsky = save_chunk(
            HYPER,
            df_test_clear_sky,
            test_chunk_counter_clearsky,
            HYPER.PATH_TO_DATA_CLIMART_CLEARSKY_TEST,
            'testing'
        )
        df_test_pristine, test_chunk_counter_pristine = save_chunk(
            HYPER,
            df_test_pristine,
            test_chunk_counter_pristine,
            HYPER.PATH_TO_DATA_CLIMART_PRISTINE_TEST,
            'testing'
        )
        
    
    ### Tell us the rations that result from our splitting rules
    n_train = (train_chunk_counter_clearsky * HYPER.CHUNK_SIZE_CLIMART) + len(df_train_clear_sky.index)
    n_val = (val_chunk_counter_clearsky * HYPER.CHUNK_SIZE_CLIMART) + len(df_val_clear_sky.index)
    n_test = (test_chunk_counter_clearsky * HYPER.CHUNK_SIZE_CLIMART) + len(df_test_clear_sky.index)
    n_total = n_train + n_val + n_test
    
    print(
        "Training data   :    {:.0%} \n".format(n_train/n_total),
        "Validation data :    {:.0%} \n".format(n_val/n_total),
        "Testing data    :    {:.0%} \n".format(n_test/n_total)
    )
    
    ### Save results of last iteration
    df_train_clear_sky, train_chunk_counter_clearsky = save_chunk(
        HYPER,
        df_train_clear_sky,
        train_chunk_counter_clearsky,
        HYPER.PATH_TO_DATA_CLIMART_CLEARSKY_TRAIN,
        'training',
        last_iteration=True    
    )
    df_train_pristine, train_chunk_counter_pristine = save_chunk(
        HYPER,
        df_train_pristine,
        train_chunk_counter_pristine,
        HYPER.PATH_TO_DATA_CLIMART_PRISTINE_TRAIN,
        'training',
        last_iteration=True    
    )
    df_val_clear_sky, val_chunk_counter_clearsky = save_chunk(
        HYPER,
        df_val_clear_sky,
        val_chunk_counter_clearsky,
        HYPER.PATH_TO_DATA_CLIMART_CLEARSKY_VAL,
        'validation',
        last_iteration=True
    )
    df_val_pristine, val_chunk_counter_pristine = save_chunk(
        HYPER,
        df_val_pristine,
        val_chunk_counter_pristine,
        HYPER.PATH_TO_DATA_CLIMART_PRISTINE_VAL,
        'validation',
        last_iteration=True
    )
    df_test_clear_sky, test_chunk_counter_clearsky = save_chunk(
        HYPER,
        df_test_clear_sky,
        test_chunk_counter_clearsky,
        HYPER.PATH_TO_DATA_CLIMART_CLEARSKY_TEST,
        'testing',
        last_iteration=True
    )
    df_test_pristine, test_chunk_counter_pristine = save_chunk(
        HYPER,
        df_test_pristine,
        test_chunk_counter_pristine,
        HYPER.PATH_TO_DATA_CLIMART_PRISTINE_TEST,
        'testing',
        last_iteration=True
    )
    
    return_df_bundle = (
        df_train_clear_sky, 
        df_train_pristine, 
        df_val_clear_sky, 
        df_val_pristine, 
        df_test_clear_sky, 
        df_test_pristine
    )
    
    return return_df_bundle
        

    
def save_chunk(
    HYPER,
    df,
    chunk_counter,
    saving_path,
    filename,
    last_iteration=False 
):

    """ """
    
    ### Save resulting data in chunks
    while len(df.index) > HYPER.CHUNK_SIZE_CLIMART or last_iteration:
        
        # increment chunk counter 
        chunk_counter += 1
        
        # create path
        path_to_saving = (
            saving_path
            + filename
            + '_{}.csv'.format(chunk_counter)
        )
        
        # shuffle
        df = df.sample(frac=1, random_state=HYPER.SEED)
        
        # save chunk
        df.iloc[:HYPER.CHUNK_SIZE_CLIMART].to_csv(path_to_saving, index=False)
        
        # delete saved chunk
        if not last_iteration:
            df = df[HYPER.CHUNK_SIZE_CLIMART:]
        
        # Must be set to exit loop on last iteration
        last_iteration = False
        
    return df, chunk_counter    
    
    
    
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
