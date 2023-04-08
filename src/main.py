import yaml
import parse_args, upload_dataset, shuffle
from selberai.data import download_dataset
from process import building_electricity, wind_farm, uber_movement, climart


if __name__ == "__main__":
  """
  Parses command line arguments and loads the configurations from config.yml
  and then executes program accordingly.
  """
  
  # parse the command line arguments
  args = parse_args.parse_arguments()
  
  # load config from yaml file
  with open("config.yml", "r") as configfile:
    config = yaml.safe_load(configfile)

  if args.upload is not None:
    upload_dataset.upload(config, args.upload)
  elif args.download is not None:
    download_dataset.download(config, args.download)
  elif args.process is not None:
    if args.process == 'BuildingElectricity':
      building_electricity.process_all_datasets(config)
    elif args.process == 'WindFarm':
      wind_farm.process_all_datasets(config)
    elif args.process == 'UberMovement':
      uber_movement.process_all_datasets(config)
    elif args.process == 'Climart':
      climart.process_all_datasets(config)
    elif args.process == 'OpenCatalyst':
      print("To do: Implement Open Catalyst dataset processing.")
  elif args.shuffle is not None:
    print("Shuffling processed {} data.".format(args.shuffle))
    config[args.shuffle]['seed'] = config['seed']
    shuffle.shuffle_data_files(config[args.shuffle])
  
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
