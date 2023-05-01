#!/bin/bash

# Check if both model path and config path arguments are provided
if [ $# -ne 2 ]; then
  echo "Usage: ./script.sh <model_path> <config_path>"
  exit 1
fi

# Assign the arguments to variables
model_path=$1
config_path=$2

# Generate datetime string for the show-dir
datetime=$(date +"%Y%m%d_%H%M%S")
show_dir="./dataset_visualization/test_result/$datetime"
mkdir -p "$show_dir"
#"./work_dirs/dataset_visualization/pipline/$datetime"
# Execute the Python command with the provided paths and show-dir
python tools/test.py "$config_path" "$model_path" --show-dir "$show_dir"
