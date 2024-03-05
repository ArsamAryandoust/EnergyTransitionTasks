import sys
import os

import yaml
import parse_args, test_dataset, analyse_dataset, shuffle 
import baseline

from data import download_data, upload_data
from process import building_electricity, wind_farm, uber_movement, climart
from process import polianna, open_catalyst


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
    resp = input('Please enter "yes" or "no"! \n'.format(args.upload))
    
    if resp == 'yes' or resp == 'y':
      upload_data.upload(config, args.upload)
    
    else:
      print('\nYou chose "no". Process interrupted!')
      exit(1)
  
  # download passed dataset
  elif args.download is not None:
    download_data.download(config, args.download)
    
  # process passed dataset
  elif args.process is not None:
    
    if args.process == 'BuildingElectricity':
      building_electricity.process_all_datasets(config, save=True)
    
    elif args.process == 'WindFarm':
      wind_farm.process_all_datasets(config, save=True)
    
    elif args.process == 'UberMovement':
      uber_movement.process_all_datasets(config, save=True)
    
    elif args.process == 'ClimART':
      climart.process_all_datasets(config, save=True)
    
    elif args.process == 'OpenCatalyst':
      open_catalyst.process_all_datasets(config, save=True)
      
    elif args.process == 'Polianna':
      polianna.process_all_datasets(config, save=True)
  
  # shuffle files of passed dataset
  elif args.shuffle is not None:
    shuffle.shuffle_data_files(args.shuffle, config)
      
  # test passed dataset
  elif args.test is not None:
    test_dataset.test(config, args.test[0], args.test[1])
    
  # analyse passed dataset
  elif args.analyse is not None:
    analyse_dataset.analyse(config, args.analyse[0], args.analyse[1])
    
  elif args.baseline is not None:
    baseline.run_baseline(config, args.baseline[0], args.baseline[1], float(args.baseline[2]) if len(args.baseline) > 2 else 1.0)
  
  print("Successfully executed all instructions!")
