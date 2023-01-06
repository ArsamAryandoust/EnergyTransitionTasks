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
        
        # decide which meteorological data to consider
        self.METEO_NAME_LIST = [
            'air_density', 'cloud_cover', 'precipitation', 'radiation_surface',
            'radiation_toa', 'snow_mass', 'snowfall', 'temperature', 'wind_speed'
        ]
        
        # splits
        self.TEST_SPLIT = 0.5
        self.TRAIN_VAL_SPLIT = 0.8
                
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
        self.PATH_TO_DATA_BUILDING_ELECTRICITY_ADD = (
            self.PATH_TO_DATA_BUILDING_ELECTRICITY
            + 'additional/'
        )
        self.PATH_TO_DATA_BUILDING_ELECTRICITY_TRAIN = (
            self.PATH_TO_DATA_BUILDING_ELECTRICITY
            + 'training/'
        )
        self.PATH_TO_DATA_BUILDING_ELECTRICITY_VAL = (
            self.PATH_TO_DATA_BUILDING_ELECTRICITY
            + 'validation/'
        )
        self.PATH_TO_DATA_BUILDING_ELECTRICITY_TEST = (
            self.PATH_TO_DATA_BUILDING_ELECTRICITY
            + 'testing/'
        )
        
        
        # create directories
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_BUILDING_ELECTRICITY)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_BUILDING_ELECTRICITY_ADD)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_BUILDING_ELECTRICITY_TRAIN)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_BUILDING_ELECTRICITY_VAL)
        HyperParameter.check_create_dir(self, self.PATH_TO_DATA_BUILDING_ELECTRICITY_TEST)
        
        
