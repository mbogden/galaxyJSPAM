#!/bin/bash

# Script to convert all Jupyter Notebook files to Python scripts in the current directory

# If no arguments are provided, assume the current directory is the target
if [ $# -eq 0 ]; then
    target="."
else
    target="$1"
fi

# if 2nd argument is provided, use it as the location of update_build_env.py
if [ $# -eq 2 ]; then
    # echo "2nd argument provided: $2"
    update_location="$2"
else
    update_location="update_build_env.py"
fi

# If not found in current directory, check utilities directory
if [ ! -f "$update_location" ]; then
    update_location="utilities/update_build_env.py"
fi

# If not found check parent/utilities directory
if [ ! -f "$update_location" ]; then
    update_location="../utilities/update_build_env.py"
fi

# exit if update file not found
if [ ! -f "$update_location" ]; then
    echo "update_build_env.py not found in '.', 'utilities', '../utilities'."
    exit 1
fi

# Find all .ipynb files in the target directory
notebooks=("$target"/*.ipynb)

# Check if any .ipynb files were found
if [ ! -e "${notebooks[0]}" ]; then
    echo "No .ipynb files found in $target."
    exit 0
fi

echo ""

# Loop through the notebooks and process them
for notebook in "${notebooks[@]}"; do

    # create name of python script
    script="${notebook%.ipynb}.py"
    echo "Converting $notebook to $script"

    # Use jupyter nbconvert to convert them to Python scripts
    # jupyter nbconvert --to script "$notebook"
    jupyter nbconvert --to=python "$notebook"

    # Run update_build_env.py to change buildEnv to False
    echo "Updating $script"
    python3 "$update_location" "$script"

    echo ""
    
done