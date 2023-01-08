import os


class HyperParameter:

    """
    Bundles a bunch of hyper parameters.
    """
    
    # Choose which data to process
    PROCESS_UBERMOVEMENT = True
    PROCESS_CLIMART = False
    PROCESS_OPENCATALYST = False
    PROCESS_BUILDINGELECTRICITY = False
    
    # Choose which data to shuffle
    SHUFFLE_UBERMOVEMENT = False
    SHUFFLE_CLIMART = False
    
    # random seed
    SEED = 3
    
    # data paths
    PATH_TO_DATA = '../data/'
    PATH_TO_DATA_RAW = PATH_TO_DATA + 'raw/'
    
    
    def check_create_dir(self, path):
    
        """ """
        
        if not os.path.isdir(path):
            os.mkdir(path)
