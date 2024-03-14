import cv2
import os
import json
import numpy as np
from skimage import measure
from pycocotools import mask
import pycocotools.mask as coco_mask
import pycocotools.mask as mask_utils
from coco_tool.generate_coco_annoation import binary_mask_to_coco_annotation
from zebra_polygon_threshold import create_mask
input_folder = "/nas/ai_image/sync_image/baidu_pan_download/栗子/鲜栗子/2022-08-03_16-57-58_0806[freshchestnut_raw]/"
output_folder = "/nas/ai_image/sync_image/baidu_pan_download/栗子/鲜栗子/marked"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(f"{output_folder}")
    os.makedirs(f"{output_folder}/Debug")
    os.makedirs(f"{output_folder}/Images")
    os.makedirs(f"{output_folder}/Annotations")

annotations = []
images = []
image_id = 1
annotation_id = 1

category_id = 1
categories = [{"id": category_id, "name": "zebra_line"}]
# Function to encode binary mask
def encode_binary_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".bmp"):
        # Read the image
        image = cv2.imread(os.path.join(input_folder, filename))
        height, width, _ = image.shape

        # Convert color space to LAB
        #lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        # Apply color threshold
        # lower_threshold = np.array([int(1.094 * 255 / 100), int(-34.913 + 128), int(-31.077 + 128)], dtype="uint8")
        # upper_threshold = np.array([int(94.115 * 255 / 100), int(26.997 + 128), int(18.403 + 128)], dtype="uint8")
        # binary_mask = cv2.inRange(lab_image, lower_threshold, upper_threshold)
        binary_mask, debug_mask= create_mask(image)
        print(f"saving debug image")
        # Save the output image
        cv2.imwrite(os.path.join(f"{output_folder}/Debug/", filename), debug_mask)
        cv2.imwrite(os.path.join(f"{output_folder}/Images/", filename), image)
        # Add image to images list
        images.append({"file_name": filename, "height": height, "width": width, "id": image_id})

        annotation = binary_mask_to_coco_annotation(binary_mask,image_id,category_id,annotation_id)
        annotations.append(annotation)

        image_id += 1
        annotation_id += 1

coco_annotations = {
    "images": images,
    "annotations": annotations,
    "categories": categories,
}



# Save the annotation file
with open(os.path.join(f"{output_folder}/Annotations", "coco_info.json"), "w") as f:
    json.dump(coco_annotations, f)