import argparse

def parse_arguments() -> argparse.Namespace:
  """ 
  Parses the command line arguments passed to the program
  """
  parser = argparse.ArgumentParser(
    prog="EnergyTransitionTasks",
    description= """ Processes raw energy transition tasks datasets. 
    We currently implement the processing of four datasets:
    
    - Building Electricity
    - Wind Farm
    - Uber Movement
    - ClimART
    - Open Catalyst
    
    An extensive explanation of all tasks and datasets is provided in our
    research article entitled: 
    
    Prediction tasks and datasets for enhancing the global energy transition
    
    """,
    epilog="Thanks for working to tackle climate change!!!"
  )
  
  parser.add_argument(
    "-upload",
    help="Pass dataset name you want to upload to Harvard Dataverse!")
  parser.add_argument(
    "-download",
    help="Pass dataset name you want to download from Harvard Dataverse!")
  parser.add_argument(
    "-process",
    help="Process datasets for Building Electricity task")
  parser.add_argument(
    "-shuffle",
    help="Pass dataset name you want to shuffle!")
  
  # parse arguments
  args = parser.parse_args()
  
  # do some checks for validity of args
  if (args.upload is None and args.download is None):
    print("\nNo download or upload operations where requested!\n")
    if args.shuffle is None:
      print("\n No data shuffling operation is requested!\n")
      if args.shuffle is None:
        print("\n No dataset has been requested to be processed!\n")
        print("Must select at least one dataset to process!")
        exit(1)
  
  return args
