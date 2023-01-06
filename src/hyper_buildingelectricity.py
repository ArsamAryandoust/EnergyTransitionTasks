from hyper import HyperParameter
import os
import random


class HyperBuildingElectricity(HyperParameter):

    """
    Bundles a bunch of hyper parameters.
    """
    
    
    def __init__(self):
    
        """ """
        
        # decide the size of prediction window in 15-min time steps
        self.PREDICTION_WINDOW = 96 # corresponds to 24h = 1 days
        
        
        # data paths
        self.PATH_TO_RAW_BUILDING_ELECTRICITY = (
            HyperParameter.PATH_TO_DATA_RAW
            + 'BuildingElectricity/profiles_400/'
        )
        self.PATH_TO_RAW_BUILDING_YEAR_PROFILES_FILE  = (
            self.PATH_TO_RAW_BUILDING_ELECTRICITY
            + 'building-year profiles/feature_scaled/2014 building-year profiles.csv'
        )
        self.PATH_TO_RAW_METEO_DATA_FOLDER  = (
            self.PATH_TO_RAW_BUILDING_ELECTRICITY
            + 'meteo data/'
        )
        self.PATH_TO_RAW_AERIAL_IMAGERY_FILE  = (
            self.PATH_TO_RAW_BUILDING_ELECTRICITY
            + 'building imagery/histogram/rgb/pixel_values.csv'
        )
        
        self.PATH_TO_DATA_BUILDING_ELECTRICITY = (
            self.PATH_TO_DATA
            + 'BuildingElectricity/'
        )
        
        
        # create directories
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_BUILDING_ELECTRICITY)
