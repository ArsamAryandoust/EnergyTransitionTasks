import parse_args
from load_config import get_config_from_yaml
from datasets import building_electricity



if __name__ == "__main__":
    """
    Parses the command line arguments and loads the configurations from config.yml
    and then executes program accordingly.
    """
    
    # get the command line arguments
    args = parse_args.parse_arguments()
    
    # get config from yaml file
    config = load_config.get_config_from_yaml()
        
    if args.building_electricity:
        print("Processing Building Electricity dataset.")
        building_electricity.process_all_datasets(config)
        
    if args.uber_movement:
        print("Processing Uber Movement dataset.")
        config = load_config.config_UM(config)
        
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
        



"""
import prep_uber_movement
import prep_open_catalyst
import prep_climart
import prep_building_electricity
import gc

# process building electricity data    
if HYPER.PROCESS_BUILDINGELECTRICITY:
    
    # tell us whats going on
    print('Processing Building Electricity data')
    
    # create hyper parameters
    HYPER_BUILDINGELECTRICITY = hyper_buildingelectricity.HyperBuildingElectricity()
    
    # import all data
    df_consumption, df_building_images, df_meteo_dict = prep_building_electricity.import_all_data(HYPER_BUILDINGELECTRICITY)

    # process building imagery
    _ = prep_building_electricity.process_building_imagery(HYPER_BUILDINGELECTRICITY, df_building_images)
    
    # process meteo and load profiles
    _, _, _ = prep_building_electricity.process_meteo_and_load_profiles(
        HYPER_BUILDINGELECTRICITY, 
        df_consumption, 
        df_meteo_dict
    )
    
    # empty memory
    del _, df_consumption, df_building_images, df_meteo_dict
    gc.collect()

# create main hyper paramter instance
HYPER = hyper.HyperParameter()


# process uber movement data
if HYPER.PROCESS_UBERMOVEMENT:

    # tell us whats going on
    print('Processing Uber Movement data')
    
    # create hyper parameters
    HYPER_UBERMOVEMENT = hyper_ubermovement.HyperUberMovement()
    
    # create geographic data
    _, _ = prep_uber_movement.process_geographic_information(HYPER_UBERMOVEMENT)
    
    # create training validation testing splits
    _, _, _ = prep_uber_movement.train_val_test_split(HYPER_UBERMOVEMENT)
    
    
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
