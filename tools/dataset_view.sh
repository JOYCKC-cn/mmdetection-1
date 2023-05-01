#!/bin/bash

# Check if the path argument is provided
if [ $# -eq 0 ]; then
  echo "Usage: ./script.sh <config path>"
  exit 1
fi

# Assign the path argument to a variable
path_arg=$1
datetime=$(date +"%Y%m%d_%H%M%S")

# Create directory using datetime string as path name
output_path="./work_dirs/dataset_visualization/pipline/$datetime"
mkdir -p "output_path $output_path"
# Execute the Python command with the path argument
python tools/analysis_tools/browse_dataset.py "$path_arg" --cfg-options "model=pipeline" --output-dir $output_path
