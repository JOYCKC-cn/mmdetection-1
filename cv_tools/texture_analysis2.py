import cv2
import numpy as np
import os
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from skimage.filters import gabor
from mahotas.features import haralick as mahotas_haralick
import pywt


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    return image


def gabor_filters(image):
    gabor_image, _ = gabor(image, frequency=0.6)
    gabor_image = cv2.normalize(gabor_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gabor_image




def local_binary_patterns(image, radius=3, n_points=24):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    lbp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return lbp


def wavelet_decomposition(image):
    coeffs = pywt.dwt2(image, 'db1')
    cA, (cH, cV, cD) = coeffs

    cA = cv2.normalize(cA, None, 0, 255, cv2.NORM_MINMAX)
    cH = cv2.normalize(cH, None, 0, 255, cv2.NORM_MINMAX)
    cV = cv2.normalize(cV, None, 0, 255, cv2.NORM_MINMAX)
    cD = cv2.normalize(cD, None, 0, 255, cv2.NORM_MINMAX)

    return np.hstack([np.vstack([cA, cH]), np.vstack([cV, cD])])


def haralick_features(image):
    image = image.astype(np.uint8)
    features = mahotas_haralick(image)
    mean_features = features.mean(axis=0)
    return mean_features


def process_images(input_dir, output_dir):
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jpg")]

    for image_path in image_paths:
        print(f"Processing {image_path}")

        preprocessed_image = preprocess_image(image_path)

        gabor_image = gabor_filters(preprocessed_image)
        lbp_image = local_binary_patterns(preprocessed_image)
        wavelet_image = wavelet_decomposition(preprocessed_image)
        haralick_text = f"Haralick Contrast: {haralick_features(preprocessed_image)[1]:.2f}"

        cv2.putText(preprocessed_image, haralick_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

        output_image = cv2.hconcat([
            preprocessed_image,
            gabor_image,
            lbp_image,
            wavelet_image
        ])

        output_file = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_file, output_image)


if __name__ == "__main__":

    input_dir = "/opt/workspace/imagedb/slice/20232828-002812-full-dry/samples/"
    output_dir = "/opt/workspace/imagedb/slice/20232828-002812-full-dry/samples_debug/"
   


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_images(input_dir, output_dir)
