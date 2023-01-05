import hyper
import prep_uber_movement
import prep_open_catalyst
import prep_climart

# create hyper paramter instance
HYPER = hyper.HyperParameter()


# process open catalyst project data
if HYPER.PROCESS_OPENCATALYST:
    _ = prep_open_catalyst.create_augment_data(HYPER)
    _, _, _ = prep_open_catalyst.train_val_test_create(HYPER)
    

# process uber movement data
if HYPER.PROCESS_UBERMOVEMENT:
    _, _ = prep_uber_movement.process_geographic_information(HYPER)
    _, _, _ = prep_uber_movement.train_val_test_split(HYPER)
    prep_uber_movement.shuffle_data_files(HYPER)
    
    
# process climart data
if HYPER.PROCESS_CLIMART:
    _, _, _, _, _, _ = prep_climart.train_val_test_split(HYPER)
    prep_climart.shuffle_data_files(HYPER)
    
    
