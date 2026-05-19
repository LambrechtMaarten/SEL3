#!/bin/bash

# Slightly changed venv script from the tutorial

# Should be ran from root folder of project 

# We create virtual environments on the $VSC_DATA storage, as JAX takes quite some space.
mkdir $VSC_DATA/venvs
# To find it easier, we create a symbolic link
ln -s $VSC_DATA/venvs ./venv

# Load the necessary modules (always load these when you work on this project)
ml load GCCcore/12.3.0
ml load Python/3.11.3-GCCcore-12.3.0
ml load FFmpeg/6.0-GCCcore-12.3.0

python --version

echo "Loaded modules"

# Create the virtual environment (cluster dependent)
cd venv
python -m venv sel3_${VSC_INSTITUTE_CLUSTER}
cd ..

echo "Created venv"

# Activate the virtual environment
source venv/sel3_${VSC_INSTITUTE_CLUSTER}/bin/activate
pip install --upgrade pip

echo "Installing packages"

# Install packages
pip install -r requirements.txt -c constraints.txt

# Uncomment if you want to be able to use jupyter lab
# python -m ipykernel install --user --name=sel3_${VSC_INSTITUTE_CLUSTER}-kernel
