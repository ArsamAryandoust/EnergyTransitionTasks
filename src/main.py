import yaml
import parse_args, upload_dataset, test_dataset, analyse_dataset, shuffle 
from selberai.data import download_data
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

  # check which instruction is passed and execute accordingly
  # upload passed dataset
  if args.upload is not None:
    # set a safety instance
    print('\nAre you sure you want to upload {}? \n'.format(args.upload))
    resp = input('Type yes or no! \n'.format(args.upload))
    if resp == 'yes' or resp == 'y':
      upload_dataset.upload(config, args.upload)
    else:
      exit(1)
  # download passed dataset
  elif args.download is not None:
    download_dataset.download(config, args.download)
  # process passed dataset
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
  # test passed dataset
  elif args.test is not None:
    test_dataset.test(config, args.test[0], args.test[1])
  # analyse passed dataset
  elif args.analyse is not None:
    analyse_dataset.analyse(config, args.analyse)
  # shuffle files of passed dataset
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
