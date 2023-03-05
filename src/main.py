import yaml

import parse_args
from datasets import building_electricity, wind_farm, uber_movement, climart
from datasets import additional

if __name__ == "__main__":
    """
    Parses the command line arguments and loads the configurations from config.yml
    and then executes program accordingly.
    """
    
    # parse the command line arguments
    args = parse_args.parse_arguments()
    
    # load config from yaml file
    with open("config.yml", "r") as configfile:
        config = yaml.safe_load(configfile)
    
    # do the processing according to command line arguments that were passed
    if args.building_electricity:
        building_electricity.process_all_datasets(config)
    if args.wind_farm:
        wind_farm.process_all_datasets(config)
    if args.uber_movement:
        uber_movement.process_all_datasets(config)
    if args.climart:
        climart.process_all_datasets(config)
    if args.open_catalyst:
        print("Processing Open Catalyst dataset.")
    if args.shuffle_UM:
        print("Shuffling processed Uber Movement data.")
        config['uber_movement']['seed'] = config['seed']
        additional.shuffle_data_files(config['uber_movement'])
        
    if args.shuffle_CA:
        print("Shuffling processed ClimArt data.")
        config['climart']['seed'] = config['seed']
        additional.shuffle_data_files(config['climart'])
    
    print("Successfully executed all instructions!")



"""
import prep_open_catalyst

# process open catalyst project data
if HYPER.PROCESS_OPENCATALYST:
    
    # tell us whats going on
    print('Processing Open Catalyst data')
    
    # create hyper parameters
    HYPER_OPENCATALYST = hyper_opencatalyst.HyperOpenCatalyst()
    
    # augment data
    _ = prep_open_catalyst.create_augment_data(HYPER_OPENCATALYST)
    
    # create train validation testing data
    _, _, _ = prep_open_catalyst.train_val_test_create(HYPER_OPENCATALYST)
    
    
"""
