#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_folder>"
    exit 1
fi

input_folder="$1"
grid_size=8

for subdir in "$input_folder"/*; do
    if [ -d "$subdir" ]; then
        python concat_grid.py "$subdir" $grid_size
    fi
done
