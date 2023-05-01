import os
import cv2
import numpy as np
# Path to the input image folder
input_folder = "/opt/workspace/imagedb/slice/palm_date_slice/zebra/20231424-001452/sliced_image"

# Path to the output image folder
output_folder = os.path.dirname(input_folder) 

# Size of each input image
img_size = (224, 224)

# Number of rows and columns in the output grid
num_rows = 10
num_cols = 10

# Get a list of all image filenames in the input folder
image_filenames = os.listdir(input_folder)

# Sort the filenames alphabetically to ensure consistent ordering
image_filenames.sort()

# Create an empty output grid image
output_grid = np.zeros((img_size[0]*num_rows, img_size[1]*num_cols, 3), dtype=np.uint8)

# Loop over the first 25 images and add them to the output grid
for i, filename in enumerate(image_filenames[:num_rows*num_cols]):
    # Load the image
    img = cv2.imread(os.path.join(input_folder, filename))

    # Resize the image to the desired size
    img = cv2.resize(img, img_size)

    # Calculate the row and column indices in the output grid
    row_idx = i // num_cols
    col_idx = i % num_cols

    # Compute the coordinates of the top-left corner of the image in the output grid
    top = row_idx * img_size[0]
    left = col_idx * img_size[1]

    # Copy the image to the output grid
    output_grid[top:top+img_size[0], left:left+img_size[1], :] = img

# Save the output grid image to the output folder
cv2.imwrite(os.path.join(output_folder, "output.jpg"), output_grid)
