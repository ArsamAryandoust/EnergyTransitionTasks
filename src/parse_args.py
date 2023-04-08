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
    help="Upload data to EnergyTransitionTasks https://dataverse.harvard.edu!")
  parser.add_argument(
    "-download",
    help="NOT implemented! Download data from https://dataverse.harvard.edu!")
  parser.add_argument(
    "-process",
    help="Process datasets for passed task")
  parser.add_argument(
    "-test",
    nargs='+'
    help="Test processed datasets for passed task")
  parser.add_argument(
    "-analyse",
    help="Analyse processed datasets for passed task")
  parser.add_argument(
    "-shuffle",
    help="Pass dataset name you want to shuffle!")
  
  # parse arguments
  args = parser.parse_args()
  
  # do some checks for validity of args
  if (args.upload is None and args.download is None):
    print("\nNo download or upload operations where requested!\n")
    if args.process is None:
      print("\nNo dataset has been requested to be processed!\n")
      if args.test is None:
        print("\nNo dataset has been requested to be tested!\n")
        if args.analyse is None:
          print("\nNo dataset has been requested to be analyzed!\n")
          if args.shuffle is None:
            print("\nNo data shuffling operation is requested!\n")
            print("Must select one of these instructions!")
            exit(1)
  
  return args
