import argparse

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
        "-building_electricity", "-BE", "-be"
        help="Process datasets for Building Electricity task",
        action="store_true"
    )
    parser.add_argument(
        "-uber_movement", "-UM", "-um",
        help="Process datasets for Uber Movement task",
        action="store_true"
    )
    parser.add_argument(
        "-climart", "-CA", "-ca",
        help="Process datasets for ClimART task",
        action="store_true"
    )
    parser.add_argument(
        "-open_catalyst", "-OC", "-oc", 
        help="Process datasets for Open Catalyst task",
        action="store_true"
    )
    
    # shuffling
    parser.add_argument(
        "--shuffle_UM", "--shuffle_um",
        help="Shuffle processed datasets for Uber Movement task",
        action="store_true"
    )
    parser.add_argument(
        "--shuffle_CA", "--shuffle_ca",
        help="Shuffle processed datasets for ClimART task",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    
    # do some checks for validity of args
    if not (
        args.building_electricity or args.uber_movement or args.open_catalyst or
        args.climart or args.shuffle_UM or args.shuffle_CA
    ):
        print("Must select at least one dataset to process or shuffle!")
        exit(1)

    return args
