import cv2
import numpy as np
import os
def dark_color_percentage(image, threshold=50):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Count the number of dark pixels
    dark_pixels = np.sum(gray_image < threshold)

    # Calculate the total number of pixels
    total_pixels = gray_image.size

    # Calculate the percentage of dark pixels
    dark_percentage = (dark_pixels / total_pixels) * 100

    return dark_percentage

def put_percentage_on_image(image, percentage):
    # Choose the font, scale, and color
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (0, 255, 0)  # Green

    # Define the text to be displayed
    text = f"Dark color percentage: {percentage:.2f}%"

    # Define the position of the text
    position = (10, 30)

    # Add the text to the image
    cv2.putText(image, text, position, font, scale, color, 2)

    return image
def preprocess(image, color_space, lower_color, upper_color, morph_operations=None, ops_kernel=[(5, 5), (5, 5)]):
    original_image=image.copy()
    # Convert the input image to the specified color space
    if color_space.lower() == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space.lower() == 'lab':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Apply color threshold
    colorthreshold_mask = cv2.inRange(image, lower_color, upper_color)

    # Apply morph operations
    if morph_operations is None:
        morph_operations = ['dilate_erode', 'erode_dilate']

    morph_mask = colorthreshold_mask.copy()

    for operation, kernel_size in zip(morph_operations, ops_kernel):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

        if operation == 'dilate_erode':
            morph_mask = cv2.dilate(morph_mask, kernel)
            morph_mask = cv2.erode(morph_mask, kernel)
        elif operation == 'erode_dilate':
            morph_mask = cv2.erode(morph_mask, kernel)
            morph_mask = cv2.dilate(morph_mask, kernel)
    morph_mask=cv2.bitwise_not(morph_mask)
    # Apply mask to the original image
    masked_image = apply_mask(original_image, morph_mask)

    # Convert masks to 3-channel images and hconcat
    colorthreshold_mask_3ch = cv2.cvtColor(colorthreshold_mask, cv2.COLOR_GRAY2BGR)
    morph_mask_3ch = cv2.cvtColor(morph_mask, cv2.COLOR_GRAY2BGR)
    hconcat_image = np.hstack((original_image, colorthreshold_mask_3ch, morph_mask_3ch, masked_image))

    return masked_image, hconcat_image


def apply_mask(image, mask):
    image_float = image.astype(np.float32) / 255.0
    mask_normalized = mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
    masked_image = cv2.multiply(image_float, mask_3ch)
    return (masked_image * 255).astype(np.uint8)


if __name__ == "__main__":
    input_dir = "/opt/workspace/imagedb/slice/20232828-002812-full-dry/samples/"
    output_dir = "/opt/workspace/imagedb/slice/20232828-002812-full-dry/samples_debug/"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    lower_color_light = np.array([51.3, 0, 0])
    upper_color_light = np.array([112.68, 255, 255])
    
    lower_color_dark = np.array([41.04, 0, 0])
    upper_color_dark = np.array([114.66, 255, 255])
    

    color_space = 'hsv'  # or 'lab'

    for img_file in os.listdir(input_dir):
        input_image = cv2.imread(os.path.join(input_dir, img_file))
        morph_operations=['erode_dilate']
        ops_kernel=[(15, 15), (1, 1)]
        dkp=dark_color_percentage(input_image,threshold=50)
        if(dkp*100>1):
            lower_color=lower_color_dark
            upper_color=upper_color_dark
        else:
            lower_color=lower_color_light
            upper_color=upper_color_light
        masked_image, hconcat_image = preprocess(input_image, color_space, lower_color, upper_color,morph_operations,ops_kernel)
        put_percentage_on_image(hconcat_image,dkp)
        cv2.imwrite(os.path.join(output_dir, img_file), hconcat_image)
