#!/bin/bash

# Function to decode URLs
urldecode() {
    printf "$(echo -n "$1" | sed 's/+/ /g;s/%\(..\)/\\x\1/g;')"
}
usage() {
    echo "Usage: $0 --image-path IMAGE_PATH [--config-path CONFIG_PATH] [--weights WEIGHTS]"
    echo ""
    echo "Required arguments:"
    echo "  --image-path IMAGE_PATH    Path to the image file (URL encoded)."
    echo ""
    echo "Optional arguments:"
    echo "  --config-path CONFIG_PATH  Path to the config file (default: /opt/workspace/mmdetection-1/work_dirs/rtmdet-tiny-ins-fullsize_single_cat/rtmdet-tiny-ins-fullsize_single_cat.py)"
    echo "  --weights WEIGHTS          Path to the weights file (default: /opt/workspace/mmdetection-1/work_dirs/rtmdet-tiny-ins-fullsize_single_cat/best_coco_segm_mAP_epoch_70.pth)"
    exit 1
}

# Set default values for config and weights
CONFIG_PATH="/opt/workspace/mmdetection-1/work_dirs/rtmdet-tiny-palmdat_ajwa/rtmdet-tiny-palmdat_ajwa.py"
WEIGHTS="/opt/workspace/mmdetection-1/work_dirs/rtmdet-tiny-palmdat_ajwa/best_coco_bbox_mAP_epoch_250.pth"

# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        --image-path)
        IMAGE_PATH="$2"
        shift # past argument
        shift # past value
        ;;
        --config-path)
        CONFIG_PATH="$2"
        shift # past argument
        shift # past value
        ;;
        --weights)
        WEIGHTS="$2"
        shift # past argument
        shift # past value
        ;;
        *)
        shift # past unknown option
        ;;
    esac
done
# Check if required parameters are set
if [ -z "$IMAGE_PATH" ]; then
    echo "Error: Missing required --image-path parameter."
    echo ""
    usage
fi

# Decode image path
DECODED_IMAGE_PATH=$(urldecode "$IMAGE_PATH")
FOLDER_NAME=$(basename "$DECODED_IMAGE_PATH")

# Create output directory with datetime and prefix
OUT_DIR="./work_dirs/inference_test/${FOLDER_NAME}_$(date +%Y-%m-%d_%H-%M-%S)"

mkdir -p "$OUT_DIR"


# Run Python script with the parsed and default parameters
python demo/image_demo.py \
"$DECODED_IMAGE_PATH" \
"$CONFIG_PATH" \
--weights "$WEIGHTS" \
--out-dir "$OUT_DIR"
