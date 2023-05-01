import numpy as np
from pycocotools import mask as maskUtils

# Assume binary mask is stored in variable 'mask'
mask = np.array([[True, False, False], [False, True, True]])

# Encode mask in RLE format
rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))

# Create COCO-style mask annotation dictionary
mask_ann = {
    'size': [mask.shape[1], mask.shape[0]],
    'counts': rle['counts'].decode('utf-8')
}

# Print mask annotation
print(mask_ann)
