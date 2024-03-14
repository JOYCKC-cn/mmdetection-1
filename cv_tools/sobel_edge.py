import cv2
import os
import numpy as np
import argparse
import timeit
import shutil


def process_images(input_dir, output_dir, dark_pixel_threshold):
    if  os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    percentages = []
    timing = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            start_time = timeit.default_timer()
            p = process_image(os.path.join(input_dir, filename), output_dir, dark_pixel_threshold)
            elapsed = timeit.default_timer() - start_time
            percentages.append(p)
            timing.append(elapsed)

    # calculate min, max, and mean for percentages
    min_val_percentages = np.min(percentages)
    max_val_percentages = np.max(percentages)
    mean_val_percentages = np.mean(percentages)

    # print them
    print("Percentages:")
    print("Minimum value: ", min_val_percentages)
    print("Maximum value: ", max_val_percentages)
    print("Mean value: ", mean_val_percentages)

    # calculate min, max, and mean for timing
    min_val_timing = np.min(timing)
    max_val_timing = np.max(timing)
    mean_val_timing = np.mean(timing)

    # print them
    print("\nTiming:")
    print("Minimum value: ", min_val_timing)
    print("Maximum value: ", max_val_timing)
    print("Mean value: ", mean_val_timing)

    print("Total count:" ,len(timing))

def process_image(image_path, output_dir, dark_pixel_threshold):
    img = cv2.imread(image_path)  # Read the image in color
    base = os.path.basename(image_path)
    filename = os.path.splitext(base)[0]

    mask = cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), np.array([52, 119, 113]), np.array([255, 255, 255]))


    
    # Apply morphological operations to clean up mask
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
   

    mask = cv2.bitwise_not(mask)
    mask = cv2.erode(mask, kernel_open, iterations = 1)
    #cv2.imwrite(os.path.join(output_dir, filename+"mask.png"), mask)
    total_pixels = np.count_nonzero(mask)

    sobely = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=5)
    sobely = cv2.normalize(sobely, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply mask to sobel image
    masked_sobely = cv2.bitwise_and(sobely, sobely, mask=mask)
    masked_sobely_blackhat = cv2.morphologyEx(masked_sobely, cv2.MORPH_BLACKHAT, kernel_open)
    mask = cv2.erode(mask, kernel_open, iterations = 1)
    masked_sobely = cv2.bitwise_and(masked_sobely_blackhat, masked_sobely_blackhat, mask=mask)

    _, binary_img = cv2.threshold(masked_sobely_blackhat, 30, 255, cv2.THRESH_BINARY)
    dark_pixels = np.count_nonzero(binary_img)
    # Perform morphological operations
    kernel = np.ones((5,5),np.uint8)
    dilated_img = cv2.dilate(binary_img, kernel, iterations = 1)
    eroded_img = cv2.erode(dilated_img, kernel, iterations = 1)

    # Find contours
    contours, _ = cv2.findContours(eroded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale image to BGR
    img_bgr = cv2.cvtColor(masked_sobely, cv2.COLOR_GRAY2BGR)
    binary_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    masked_sobely_blackhat = cv2.cvtColor(masked_sobely_blackhat, cv2.COLOR_GRAY2BGR)
    # Draw contours on the image
    cv2.drawContours(img_bgr, contours, -1, (0,255,0), 2)
    text_canvas = np.zeros_like(img)
    # Calculate the area of each contour and annotate the image
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        #cv2.putText(text_canvas, f"Area {i}: {area}", (10, 30+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)    
    # # Create canvas and write text
   
    
    percentage = dark_pixels/total_pixels*100
    cv2.putText(text_canvas, f"Total pixels: {total_pixels}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(text_canvas, f"white pixels: {dark_pixels}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(text_canvas, f"Percentage: {percentage:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Concatenate images and save
    final_image = np.hstack((img, binary_img,img_bgr,masked_sobely_blackhat,text_canvas))
    
    cv2.imwrite(os.path.join(output_dir, f"{percentage:.2f}_"+filename + ".png"), img)
    return percentage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/opt/workspace/imagedb/train_cls_old_collection/deform_score_sample/")
    parser.add_argument("--output_dir", default="/opt/images/deform_score/")
    parser.add_argument("--dark_pixel_threshold", type=int, default=150)
    args = parser.parse_args()
    
    
    process_images(args.input_dir, args.output_dir, args.dark_pixel_threshold)
