import os
import cv2
import numpy as np
from skimage import feature
from skimage import filters
from skimage.feature import greycomatrix, greycoprops

def preprocess_image(image_path, resize_dim=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, resize_dim)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)
    return image, blurred_image

def local_binary_patterns(image, num_points=64, radius=3):
    lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform")
    return lbp

from scipy import ndimage

def gabor_filters(image, frequencies=[0.1, 0.2], thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    gabor_images = []
    for frequency in frequencies:
        for theta in thetas:
            gabor_filter = filters.gabor_kernel(frequency=frequency, theta=theta)
            gabor_image = ndimage.convolve(image, gabor_filter, mode='reflect', cval=0)
            gabor_images.append(np.abs(gabor_image))  # Take the magnitude of the complex image
    return np.mean(gabor_images, axis=0)

# def lpq(image):
#     lpq_feature = mahotas.features.lpq(image, windowsize=3)
#     return lpq_feature

def haralick_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    image = image.astype(np.uint8)
    glcm = greycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast').mean()
    return contrast
def canny_edge(blurred):

    canny = cv2.Canny(blurred, 30, 150)
    return canny
def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_path in image_paths:
        original_image, preprocessed_image = preprocess_image(image_path)
        
        # Apply Local Binary Patterns
        lbp_image = local_binary_patterns(preprocessed_image)
    
        # # Apply Gabor Filters
        # gabor_image = gabor_filters(preprocessed_image)
        # canny_edge_img = canny_edge(preprocessed_image)
        # # Apply Haralick Features (not an image)
        # haralick_contrast = haralick_features(preprocessed_image)
        # print(f"Haralick contrast for {os.path.basename(image_path)}: {haralick_contrast:.2f}")

        # # Write Haralick contrast on the original image
        # cv2.putText(original_image, f"Haralick contrast: {haralick_contrast:.2f}", (0, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Normalize LBP and Gabor images
        lbp_image = cv2.normalize(lbp_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # gabor_image = cv2.normalize(gabor_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # # lpq_image = cv2.normalize(lpq_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # # Convert LBP and Gabor images to 3-channel images
        lbp_image_color = cv2.cvtColor(lbp_image, cv2.COLOR_GRAY2BGR)
        # gabor_image_color = cv2.cvtColor(gabor_image, cv2.COLOR_GRAY2BGR)
        # canny_edge_img=cv2.cvtColor(canny_edge_img, cv2.COLOR_GRAY2BGR)
        # # lpq_image_color = cv2.cvtColor(lpq_image, cv2.COLOR_GRAY2BGR)
        # Horizontally concatenate the original image with the LBP and Gabor images
        #, gabor_image_color,canny_edge_img

        # Detect edges using Canny
        edges = cv2.Canny(lbp_image, 50, 150, apertureSize=3)

        # Normalize the edges to have values in the range [0, 1]
        mask_normalized = edges / 255

        # Convert the mask to a 3-channel image
        mask_normalized_3ch = np.stack([mask_normalized] * 3, axis=-1).astype(np.float32)
        original_image_float = original_image.astype(np.float32)
        # Multiply the original image by the mask
        masked_image = cv2.multiply(original_image_float, mask_normalized_3ch)
        # Convert the masked image back to uint8
        masked_image = masked_image.astype(np.uint8)

        hconcat_image = cv2.hconcat([original_image,cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR), lbp_image_color,masked_image])

        # Save the concatenated image
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_image_path, hconcat_image)


if __name__ == "__main__":
    input_dir = "/opt/workspace/imagedb/train_cls_old_collection/deform_score_sample/"
    output_dir = "/opt/images/deform_score/"
    process_images(input_dir, output_dir)
