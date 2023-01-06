from hyper import HyperParameter
import os
import random


class HyperBuildingElectricity(HyperParameter):

    """
    Bundles a bunch of hyper parameters.
    """
    
    
    def __init__(self):
    
        """ Set some paths by reading folders """
        
        # data paths
        self.PATH_TO_BUILDING_YEAR_PROFILES_FILE  = (
            HyperParameter.PATH_TO_DATA_RAW
            + 'BuildingElectricity/profiles_400/building-year_profiles/'
            + 'feature_scaled/2014 building-year profiles.csv'
        )
        self.PATH_TO_METEO_DATA_FOLDER  = (
            HyperParameter.PATH_TO_DATA_RAW
            + 'BuildingElectricity/profiles_400/meteo data/'
        )
        self.PATH_TO_AERIAL_IMAGERY_FOLDER  = (
            HyperParameter.PATH_TO_DATA_RAW
            + 'BuildingElectricity/profiles_400/building imagery/histogram/rgb/'
        )
        
