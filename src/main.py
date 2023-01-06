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
    df_consumption, df_building_images, df_meteo_dict = import_all_data(HYPER)

    # process building imagery
    _ = process_building_imagery(HYPER, df_building_images)
    
    # process meteo and load profiles
    _ = process_meteo_and_load_profiles(
        HYPER, 
        df_consumption, 
        df_meteo_dict
    )
    
    # empty memory
    del df_consumption, df_building_images, df_meteo_dict
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
    
    
