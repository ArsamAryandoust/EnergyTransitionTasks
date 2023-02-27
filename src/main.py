import yaml

import parse_args
from datasets import building_electricity, uber_movement


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
    if args.uber_movement:
        uber_movement.process_all_datasets(config)
    if args.climart:
        print("Processing ClimArt dataset.")
        config = load_config.config_CA(config)
    if args.open_catalyst:
        print("Processing Open Catalyst dataset.")
        config = load_config.config_OC(config)
    if args.shuffle_UM:
        print("Shuffling processed Uber Movement data.")
    if args.shuffle_CA:
        print("Shuffling processed ClimArt data.")
    
    
    print("Successfully executed all instructions!")



"""
import prep_open_catalyst
import prep_climart


    
# process climart data
if HYPER.PROCESS_CLIMART:

    # tell us whats going on
    print('Processing ClimART data')

    # create hyper parameters
    HYPER_CLIMART = hyper_climart.HyperClimart()
    
    # create train validation testing splits
    _, _, _, _, _, _ = prep_climart.train_val_test_split(HYPER_CLIMART)
    

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
    

    
# Shuffle Uber Movement data
if HYPER.SHUFFLE_UBERMOVEMENT:

    # tell us whats going on
    print('Shuffling Uber Movement data.')
    
    # create hyper parameters
    if not HYPER.PROCESS_UBERMOVEMENT:
        HYPER_UBERMOVEMENT = hyper_ubermovement.HyperUberMovement()
    
    # shuffle
    prep_uber_movement.shuffle_data_files(HYPER_UBERMOVEMENT)
    
    
# Shuffle climart data
if HYPER.SHUFFLE_CLIMART:

    # tell us whats going on
    print('Shuffling Climart data.')
    
    # create hyper parameters
    if not HYPER.PROCESS_CLIMART:
        HYPER_CLIMART = hyper_climart.HyperClimart()
    
    # shuffle
    prep_climart.shuffle_data_files(HYPER_CLIMART)
    
"""
