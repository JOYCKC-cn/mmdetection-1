#!/bin/bash

# Check if download_source argument is provided
if [ $# -eq 0 ]; then
  echo "Usage: ./script.sh <download_source> [split_ratio]"
  exit 1
fi
data_root_path="/opt/image_store1/pd_detect_mask_dataset/"
# Assign the arguments to variables
download_source=$1
split_ratio=${2:-0.8}  # Default value is 0.8 if split_ratio is not provided

# Generate datetime string for temp file name
datetime=$(date +"%Y%m%d_%H%M%S")

# Download the file using wget and rename it with the temp file name
temp_file="$datetime.tmp.zip"
wget "$download_source" -O "$temp_file"

# Unzip the downloaded file
unzip "$temp_file" -d $datetime

find_path="Dataset*"
unziped_path=$(ls "$datetime")
echo $unziped_path
echo "$datetime/$unziped_path/Annotations/coco_info.json"
mv "$datetime/$unziped_path" "$data_root_path"

# Run the Python command with the split_ratio and unziped_path
python tools/cocosplit.py --having-annotations --multi-class -s "$split_ratio" "$data_root_path/$datetime/$unziped_path/Annotations/coco_info.json" "$data_root_path/$datetime/$unziped_path/Annotations/train.json" "$data_root_path/$datetime/$unziped_path/Annotations/test.json"
echo $(ls "$datetime/$unziped_path/Annotations/")
# Remove the temp file
rm "$temp_file"
