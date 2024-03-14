import os
import cv2
import numpy as np

def preprocess_image(image_path, resize_dim=(300, 300)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, resize_dim)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return image, blurred_image

def detect_edges(image, lower_threshold=100, upper_threshold=200):
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges

def dilate_edges(edges, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=iterations)
    return dilated_edges

def count_edges(dilated_edges):
    edge_count = np.count_nonzero(dilated_edges)
    return edge_count

def normalize_count(edge_count, max_count):
    score = edge_count / max_count
    return score

def quantify_wrinkles(image_path, max_count=10000):
    original_image, preprocessed_image = preprocess_image(image_path)
    edges = detect_edges(preprocessed_image)
    dilated_edges = dilate_edges(edges)
    edge_count = count_edges(dilated_edges)
    score = normalize_count(edge_count, max_count)
    return score, original_image, dilated_edges

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_path in image_paths:
        score, original_image, dilated_edges = quantify_wrinkles(image_path)
        print(f"{os.path.basename(image_path)} Wrinkle score: {score:.2f}")

        # Convert dilated_edges to 3-channel image
        dilated_edges_color = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)

        # Horizontally concatenate the original image with the dilated edge image
        hconcat_image = cv2.hconcat([original_image, dilated_edges_color])

        # Save the concatenated image
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_image_path, hconcat_image)

if __name__ == "__main__":
    input_dir = "/opt/workspace/imagedb/train_cls_old_collection/deform_score_sample/"
    output_dir = "/opt/images/deform_score/"
    process_images(input_dir, output_dir)

