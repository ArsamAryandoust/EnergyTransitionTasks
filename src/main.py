import hyper
import hyper_ubermovement
import hyper_climart
import hyper_opencatalyst
import prep_uber_movement
import prep_open_catalyst
import prep_climart

# create main hyper paramter instance
HYPER = hyper.HyperParameter()

# process open catalyst project data
if hyper.PROCESS_OPENCATALYST:
    
    # create hyper parameters
    HYPER_OPENCATALYST = hyper_opencatalyst.HyperOpenCatalyst()
    
    # augment data
    _ = prep_open_catalyst.create_augment_data(HYPER_OPENCATALYST)
    
    # create train validation testing data
    _, _, _ = prep_open_catalyst.train_val_test_create(HYPER_OPENCATALYST)
    

# process uber movement data
if HYPER.PROCESS_UBERMOVEMENT:

    # create hyper parameters
    HYPER_UBERMOVEMENT = hyper_ubermovement.HyperUberMovement()
    
    # create geographic data
    _, _ = prep_uber_movement.process_geographic_information(HYPER_UBERMOVEMENT)
    
    # create training validation testing splits
    _, _, _ = prep_uber_movement.train_val_test_split(HYPER_UBERMOVEMENT)
    
    
# process climart data
if HYPER.PROCESS_CLIMART:

    # create hyper parameters
    HYPER_CLIMART = hyper_climart.HyperClimart()
    
    # create train validation testing splits
    _, _, _, _, _, _ = prep_climart.train_val_test_split(HYPER_CLIMART)
    
    
# Shuffle Uber Movement data
if HYPER.SHUFFLE_UBERMOVEMENT:

    # create hyper parameters
    if not HYPER.PROCESS_UBERMOVEMENT:
        HYPER_UBERMOVEMENT = hyper_ubermovement.HyperUberMovement()
    
    # shuffle
    prep_uber_movement.shuffle_data_files(HYPER_UBERMOVEMENT)
    
    
# Shuffle climart data
if HYPER.SHUFFLE_CLIMART:

    # create hyper parameters
    if not HYPER.PROCESS_CLIMART:
        HYPER_CLIMART = hyper_climart.HyperClimart()
    
    # shuffle
    prep_climart.shuffle_data_files(HYPER_CLIMART)
    
    
