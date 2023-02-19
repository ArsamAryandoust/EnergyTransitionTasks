"""
import hyper
import hyper_ubermovement
import hyper_climart
import hyper_opencatalyst
import hyper_buildingelectricity
import prep_uber_movement
import prep_open_catalyst
import prep_climart
import prep_building_electricity
import gc
"""


import argparse
import yaml

import augment_config


def parse_arguments() -> argparse.Namespace:
    """ 
    Parses the command line arguments passed to the program
    """
    parser = argparse.ArgumentParser(
        prog="EnergyTransitionTasks",
        description= """ Processes raw energy transition tasks datasets. We currently 
        implement the processing of four datasets:
        
        - Building Electricity
        - Uber Movement
        - ClimART
        - Open catalyst
        
        An extensive explanation of all tasks and datasets is provided in our
        research article entitled: 
        
        Prediction tasks and datasets for enhancing the global energy transition
        
        """,
        epilog="Thanks for working to tackle climate change! The world needs more like you!"
    )
    
    # processing
    parser.add_argument(
        "-building_electricity", 
        help="Process datasets for Building Electricity prediciton task",
        action="store_true"
    )
    parser.add_argument(
        "-uber_movement", 
        help="Process datasets for Uber Movement prediciton task",
        action="store_true"
    )
    parser.add_argument(
        "-climart", 
        help="Process datasets for ClimART prediciton task",
        action="store_true"
    )
    parser.add_argument(
        "-open_catalyst", 
        help="Process datasets for Open Catalyst prediciton task",
        action="store_true"
    )
    
    # shuffling
    parser.add_argument(
        "--shuffle_UM", 
        help="Shuffle processed datasets for Uber Movement task",
        action="store_true"
    )
    parser.add_argument(
        "--shuffle_CA", 
        help="Shuffle processed datasets for ClimART task",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    
    # do some checks for validity of args
    if not (
        args.building_electricity or args.uber_movement or args.open_catalyst or
        args.shuffle_UM or args.shuffle_CA
    ):
        print("Must select at least one dataset to process or shuffle!")
        exit(1)

    
    return args



if __name__ == "__main__":
    """
    Internal note: executing main.py inside this prevents ability to execute main.py 
    from other part of program.
    """
    
    # get the command line arguments
    args = parse_arguments()
    
    # get config from yaml file
    with open("config.yml", "r") as configfile:
        config = yaml.safe_load(configfile)
        
        
    if args.building_electricity:
        print("Processing Building Electricity dataset.")
        config = augment_config.config_BE(config)
        
        
        
    if args.uber_movement:
        print("Processing Uber Movement dataset.")
        config = augment_config.config_UM(config)
        
    
    

"""
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
