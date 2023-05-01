import os
import cv2
import numpy as np

def process_images(folder_path, dest_path, debug_path):
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        # 1. Read the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # 2. Get the shape of the image
        height, width, _ = image.shape

        # 3. Determine total columns based on image width
        if width > 1800:
            total_columns = 8
        else:
            total_columns = 6
        total_rows = 4

        # 4. Calculate single row and column width and height
        single_width = width // total_columns
        single_height = height // total_rows

        # Draw dividing lines in red and save to debug_path
        debug_image = image.copy()
        for row in range(1, total_rows):
            for x in range(0, width, 2):
                debug_image[row * single_height, x] = (0, 0, 255)
        for col in range(1, total_columns):
            for y in range(0, height, 2):
                debug_image[y, col * single_width] = (0, 0, 255)

        debug_output_filename = f"{filename}_debug.png"
        debug_output_path = os.path.join(debug_path, debug_output_filename)
        cv2.imwrite(debug_output_path, debug_image)

        # 5. Slice image and 6. Store sliced images with coordinates
        for row in range(0, total_rows, 2):
            for col in range(0, total_columns, 2):
                y1, y2 = row * single_height, (row + 2) * single_height
                x1, x2 = col * single_width, (col + 2) * single_width

                sliced_image = image[y1:y2, x1:x2]

                # Save the sliced image
                output_filename = f"{filename}_{row}_{col}_{total_rows}_{total_columns}.png"
                output_path = os.path.join(dest_path, output_filename)
                cv2.imwrite(output_path, sliced_image)

if __name__ == "__main__":
    base_path="/opt/images/cross_slice"
    folder_path = f"{base_path}/src/"
    dst_path = f"{base_path}/dst/"
    debug_path = f"{base_path}/debug/"
    
    os.makedirs(dst_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)

    process_images(folder_path, dst_path, debug_path)
