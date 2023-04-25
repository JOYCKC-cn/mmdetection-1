import cv2
import numpy as np
from pycocotools import mask as coco_mask
def binary_mask_to_coco_annotation(binary_mask, image_id, category_id, annotation_id):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []

    for contour in contours:
        # Convert contour to a polygon format
        contour = contour.flatten().tolist()
        # Check if the polygon has more than 2 points (minimum for a valid polygon)
        if len(contour) > 2:
            segmentation.append(contour)

    # Calculate the bounding box
    x, y, w, h = cv2.boundingRect(binary_mask)

    # Calculate the area
    area = int(cv2.countNonZero(binary_mask))


    # Convert binary mask to a list
    binary_mask_list = binary_mask.astype(int).tolist()

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "bbox": [x, y, w, h],
        "category_id": category_id,
        "image_id": image_id,
        "id": annotation_id,
        "mask": binary_mask_list
    }

    return annotation

