#!/bin/bash

###
# This script serves for running code on MIT's Supercloud
###

# Loading the required modules 
source /etc/profile
module load anaconda/Python-ML-2023b 

# Run main script
python src/main.py
