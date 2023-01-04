import os
import random
from ase import io

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
