import os
import random
from ase import io
import math
import pandas as pd
import numpy as np
from tqdm import tqdm


def create_dataset_df(
    HYPER, 
    path_to_dataset_folder,
    path_to_saving_folder,
    filename_saving
):

    """ """
    
    # get a list of all files on a particular path, here training data
    file_list = os.listdir(path_to_dataset_folder)
    
    # drop .txt files
    file_list = [element for element in file_list if '.txt' not in element]

    # shorten file_list for tests
    #file_list = file_list[:100]
    
    # determine how many structures/datapoints per file you want to load
    n_datapoints_per_file = 5000
    n_sample = math.floor(HYPER.SUBSAMPLE_OPENCATALYST * n_datapoints_per_file)
    import_index_str = ':{}'.format(n_sample)
    
    # declare empty dataframe
    df_dataset = pd.DataFrame()
    
    # set chunk counter for data saving to zero
    chunk_counter = 0
    
    # tell us whats going on
    print('Processing {} data'.format(filename_saving))
    
    # create progress bar
    pbar = tqdm(total=len(file_list))
    
    # iterate over all filenames
    for filename in file_list:
        
        # create path to iterated file
        path_to_file = path_to_dataset_folder + filename
        
        # import dataset
        atoms_object_list = io.read(path_to_file, index=import_index_str)
        
        # iterate over all atoms objects. Each is a datapoint/atom-structure
        for atoms_object in atoms_object_list:
            
            # get number of atoms in structure
            n_atoms = atoms_object.get_global_number_of_atoms()
            
            # get structure atom symbols as string
            symbols_atoms = atoms_object.symbols.__str__()
            
            # get structure energy we want to predict (label)
            energy = atoms_object.info['energy']
    
            # get positions (features)
            positions = atoms_object.positions
            
            # get forces (labels)
            forces = atoms_object.get_forces()
            
            # flatten coordiantes position and forces coordinates in 'C' order
            positions_array = positions.flatten()
            forces_array = positions.flatten()
            
            # create value dict for first entries
            value_dict = {
                'n_atoms': n_atoms,
                'symbols_atoms': symbols_atoms,
                'energy': energy
            }
            # create value list for dataframe
            value_array = np.concatenate(
                (
                    positions_array,
                    forces_array
                )
            )
            
            # reshape values array
            value_array = value_array.reshape(-1, len(value_array))
            
            # create column names for dataframe
            cols_list = []
            for i in range(n_atoms):
                col_name_x = 'pos_x{}'.format(i)
                col_name_y = 'pos_y{}'.format(i)
                col_name_z = 'pos_z{}'.format(i)
                cols_list.append(col_name_x)
                cols_list.append(col_name_y)
                cols_list.append(col_name_z)
            for i in range(n_atoms):
                col_name_x = 'force_x{}'.format(i)
                col_name_y = 'force_y{}'.format(i)
                col_name_z = 'force_z{}'.format(i)
                cols_list.append(col_name_x)
                cols_list.append(col_name_y)
                cols_list.append(col_name_z)
            
            # create a first part of datapoint consisting of value dictionary
            df_datapoint_part1 = pd.DataFrame(value_dict, index=[0])
            
            # create a second part of datapoint consisting of value array
            df_datapoint_part2 = pd.DataFrame(value_array, columns=cols_list)
            
            # merge both datapoint parts into a complete datapoint row as DataFrame
            df_datapoint = pd.concat([df_datapoint_part1, df_datapoint_part2], axis=1)
            
            # concatenate datapoint to existing dataset
            df_dataset = pd.concat([df_dataset, df_datapoint])
            
            
        # save dataset chunk after importing data of this data file
        df_dataset, chunk_counter = save_chunk(
            HYPER,
            df_dataset,
            chunk_counter,
            path_to_saving_folder,
            filename_saving
        )
        
        # update progress bar
        pbar.update(1)
            
            
    # save remaining dataset chunk after importing data of all data files
    df_dataset, chunk_counter = save_chunk(
        HYPER,
        df_dataset,
        chunk_counter,
        path_to_saving_folder,
        filename_saving,
        last_iteration=True 
    )
    
    # close progress bar
    pbar.close()
    
    return df_dataset



def save_chunk(
    HYPER,
    df,
    chunk_counter,
    path_to_saving_folder,
    filename_saving,
    last_iteration=False 
):

    """ """
    
    ### Save resulting data in chunks
    while len(df.index) > HYPER.CHUNK_SIZE_OPENCATALYST or last_iteration:
        
        # increment chunk counter 
        chunk_counter += 1
        
        # create path
        saving_path = (
            path_to_saving_folder
            + filename_saving
            + '_{}.csv'.format(chunk_counter)
        )
        
        # shuffle
        df = df.sample(frac=1, random_state=HYPER.SEED)
        
        # save chunk
        df.iloc[:HYPER.CHUNK_SIZE_OPENCATALYST].dropna(axis=1, how='all').to_csv(saving_path, index=False)
        
        # delete saved chunk
        if not last_iteration:
            df = df[HYPER.CHUNK_SIZE_OPENCATALYST:].dropna(axis=1, how='all')
        
        # Must be set to exit loop on last iteration
        last_iteration = False
        
    return df, chunk_counter



def process_raw_data(HYPER):

    """ """
    
    # generate training dataset
    df_training = create_dataset_df(
        HYPER, 
        HYPER.PATH_TO_DATA_RAW_OPENCATALYST_S2EF_TRAIN,
        HYPER.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TRAIN,
        'training'
    )
    
    # generate validation dataset from in distribution validation data
    df_validation = create_dataset_df(
        HYPER, 
        HYPER.PATH_TO_DATA_RAW_OPENCATALYST_S2EF_VAL_ID,
        HYPER.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_VAL,
        'validation'
    )
    
    # generate testing dataset from a constellation of out of distribution datasets
    # in terms of catalysts and adsorbates
    df_testing_1 = create_dataset_df(
        HYPER, 
        HYPER.PATH_TO_DATA_RAW_OPENCATALYST_S2EF_VAL_OOD_BOTH,
        HYPER.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TEST,
        'testing_ood_both'
    )
    df_testing_2 = create_dataset_df(
        HYPER, 
        HYPER.PATH_TO_DATA_RAW_OPENCATALYST_S2EF_VAL_OOD_CAT,
        HYPER.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TEST,
        'testing_ood_cat'
    )
    df_testing_3 = create_dataset_df(
        HYPER, 
        HYPER.PATH_TO_DATA_RAW_OPENCATALYST_S2EF_VAL_OOD_ADS,
        HYPER.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_TEST,
        'testing_ood_ads'
    )
    
    # concatenate testing datasets
    df_testing = pd.concat([df_testing_1, df_testing_2, df_testing_3])

    
    return df_training, df_validation, df_testing

def create_augment_data(HYPER):

    """ """

    # load dataset from raw .csv file
    df_periodic_table = pd.read_csv(HYPER.PATH_TO_DATA_RAW_OPENCATALYST_PTE)
    
    # create saving path
    path_to_saving = HYPER.PATH_TO_DATA_OPENCATALYST_OC20_S2EF_ADD + 'periodic_table.csv'
    
    # save dataframe on new path
    df_periodic_table.to_csv(path_to_saving, index=False)
    
    return df_periodic_table
    

def import_raw_data_samples(HYPER):

    """ """
    
    # get a list of all files on a particular path, here training data
    file_list = os.listdir(HYPER.PATH_TO_DATA_RAW_OPENCATALYST_S2EF_TRAIN)
    
    # take the first file in the list as sample. We want a .extxyz file.
    sample_file_extxyz_xz = file_list[0]
    
    # Show us which file we picked
    print('Importing sample file {} \n'.format(sample_file_extxyz_xz))
    
    # create full path to sample file
    path_to_file = HYPER.PATH_TO_DATA_RAW_OPENCATALYST_S2EF_TRAIN + sample_file_extxyz_xz
    
    # import sample file
    sample_dataset = io.read(path_to_file, index=':95')
    
    # tell us how many data points sample data contains
    print(
        'The sampled dataset contains {} data points,'.format(len(sample_dataset)),
        'each consisting of a constellation of atoms.'
    )
    
    # set the first entry as sample datapoint
    sample_datapoint = sample_dataset[0]
    
    return sample_dataset, sample_datapoint



def print_features_of_datapoint(sample_datapoint):

    """ """
    
    print('\n A single datapoint contains the following properties.')
    print('\n Global number of atoms:\n', sample_datapoint.get_global_number_of_atoms())
    print('\n Chemical formula:\n', sample_datapoint.get_chemical_formula())
    print('\n Symbols concise:\n', sample_datapoint.symbols)
    print('\n Energy:\n', sample_datapoint.info['energy'])
    print('\n Free energy:\n', sample_datapoint.info['free_energy'])
    print('\n Volume:\n', sample_datapoint.get_volume())
    print('\n Center of mass:\n', sample_datapoint.get_center_of_mass())
    print('\n Periodic boundary condition (pbc):\n', sample_datapoint.pbc)
    print('\n Cell:\n', sample_datapoint.cell)
    print('\n Symbols extensive:\n', sample_datapoint.get_chemical_symbols())
    print('\n Atomic number in periodic table:\n', sample_datapoint.numbers)
    print('\n Masses:\n', sample_datapoint.get_masses())
    print('\n Tags:\n', sample_datapoint.get_tags())
    print('\n Positions:\n', sample_datapoint.positions)
    print('\n Forces with applied constraints (Tags):\n', sample_datapoint.get_forces())
    
    """
    print('\n Cell reciprocal:\n', sample_datapoint.cell.reciprocal())
    print('\n Cell angles:\n', sample_datapoint.cell.cellpar())
    print('\n Scaled positions:\n', sample_datapoint.get_scaled_positions())
    print('\n Forces raw:\n', sample_datapoint.get_forces(apply_constraint=False))
    print('\n Constraints:\n', sample_datapoint.constraints)
    """
