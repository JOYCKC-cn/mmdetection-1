import cv2
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import os
def preprocess_image(image_path, resize_dim=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, resize_dim)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)
    return image, blurred_image

def ridge_detection(image, sigma=1):
    hessian_matrices = hessian_matrix(image, sigma)
    i1, i2 = hessian_matrix_eigvals(hessian_matrices)
    ridge_mask_max = i1 > 0
    ridge_mask_min = i2 > 0
    return ridge_mask_max,ridge_mask_min


def process_images(source_dir, output_dir):
    for img_file in os.listdir(source_dir):
        # Read the input image and convert it to grayscale
        #input_image = cv2.imread(os.path.join(source_dir, img_file))
        input_image, blurred_image = preprocess_image(os.path.join(source_dir, img_file))
        # Apply ridge detection
        ridge_mask_max,ridge_mask_min = ridge_detection(blurred_image, sigma=1)

        # Convert the mask to a 3-channel image
        ridge_mask_3ch_max = np.stack([ridge_mask_max] * 3, axis=-1).astype(np.float32)
        ridge_mask_3ch_min = np.stack([ridge_mask_min] * 3, axis=-1).astype(np.float32)

        # Multiply the original image by the mask
        original_image_float = input_image.astype(np.float32) / 255.0
        masked_image_max = cv2.multiply(original_image_float, ridge_mask_3ch_max)
        masked_image_min = cv2.multiply(original_image_float, ridge_mask_3ch_min)

        # Display the original image and the masked image side by side
        hconcat_image = np.hstack((original_image_float, masked_image_max,masked_image_min))
        cv2.imwrite(os.path.join(output_dir, img_file), hconcat_image * 255)

if __name__ == "__main__":
    input_dir = "/opt/workspace/imagedb/slice/20232828-002812-full-dry/samples/"
    output_dir = "/opt/workspace/imagedb/slice/20232828-002812-full-dry/samples_debug/"
    process_images(input_dir, output_dir)

