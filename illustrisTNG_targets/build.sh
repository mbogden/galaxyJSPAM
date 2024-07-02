#!/bin/bash

# Script to convert all Jupyter Notebook files to Python scripts in the current directory

# Find all .ipynb files in the current directory and loop through them
for notebook in *.ipynb; do
    # Use jupyter nbconvert to convert them to Python scripts
    jupyter nbconvert --to script "$notebook"
    echo "Converted $notebook to Python script."
done

echo "All notebooks have been converted."
