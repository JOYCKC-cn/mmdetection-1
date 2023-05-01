import cv2
import os
import json
import numpy as np
from skimage import measure
from pycocotools import mask
import pycocotools.mask as coco_mask
from zebra_polygon_threshold import create_mask

input_img="/opt/workspace/imagedb/slice/palm_date_slice/zebra/20231424-001452/sliced_image/46609-21-11-07 19-17-31-072_0_sliced_image_r3_c4.jpg"
I=cv2.imread(input_img)
# I = np.ones((3, 3, 3), dtype=np.uint8) * 255

mk,masked_img = create_mask(I)
print(I.shape)


print(np.nonzero(mk_3ch))
masked_img2 = cv2.bitwise_and(I, mk_3ch)

cv2.imwrite("/tmp/tmp_3.jpg",masked_img)
cv2.imwrite("/tmp/tmp3.jpg",masked_img2)

# import numpy as np
# import cv2

# original_img = np.full((3, 3, 3), 255, dtype=np.uint8)
# lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2Lab)

# # Adjust LAB values to match MATLAB
# lab = lab.astype(np.float32)
# lab[:, :, 0] = lab[:, :, 0] * (100 / 255)
# lab[:, :, 1] = lab[:, :, 1] - 128
# lab[:, :, 2] = lab[:, :, 2] - 128

# print("lab:", lab)

