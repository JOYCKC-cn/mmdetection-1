import cv2
import sys
import os
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


def read_and_resize_image(image_path, original_size, index):
    image = cv2.imread(image_path)
    if image.shape[:2] != original_size:
        image = cv2.resize(image, original_size)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text = str(index)
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (original_size[0] - text_size[0]) // 2
    text_y = text_size[1] + 10

    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 255), font_thickness)
    return image

def create_concatenated_image(image_list, grid_size_w, grid_size_h, original_size, output_name):
    result_image = np.zeros((grid_size_h * original_size[0], grid_size_w * original_size[1], 3), np.uint8)
    for i in range(grid_size_h):
        for j in range(grid_size_w):
            index = i * grid_size_w + j
            if index < len(image_list):
                image = read_and_resize_image(image_list[index], original_size, index)
            else:
                image = np.ones((*original_size[::-1], 3), np.uint8) * 255
            result_image[i * original_size[0]: (i+1) * original_size[0], j * original_size[1]: (j+1) * original_size[1]] = image

    result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(result_image_pil)

    # Load a font that supports Chinese characters (e.g., simhei.ttf)
    font = ImageFont.truetype("NotoSansSC-Light.otf", 30)
    text = output_name.split('.')[0]
    text_size = draw.textsize(text, font=font)
    text_x = (result_image.shape[1] - text_size[0]) // 2
    text_y = result_image.shape[0] - text_size[1] - 10

    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 0))
    result_image = cv2.cvtColor(np.array(result_image_pil), cv2.COLOR_RGB2BGR)
    return result_image




def main(input_path, grid_size):
    image_list = [os.path.join(input_path, img) for img in os.listdir(input_path) if img.endswith(('.png', '.jpg', '.jpeg'))][:grid_size * grid_size]
    image_list = sorted(image_list)

    if not image_list:
        print("No image files found in the input directory.")
        return

    original_image_size = cv2.imread(image_list[0]).shape[:2]

    

    output_folder_name = os.path.basename(os.path.normpath(input_path))
    output_path_dirname = os.path.dirname(os.path.normpath(input_path))
    output_filename = f"{output_folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    output_path = os.path.join(output_path_dirname, output_filename)
    concatenated_image = create_concatenated_image(image_list,8,8, original_image_size,output_folder_name)
    cv2.imwrite(output_path, concatenated_image)
    print(f"Concatenated image saved at: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {__file__} <input_path> <grid_size>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    grid_size = int(sys.argv[2])

    print(f"starting processing with input_path:{input_path} ")
    main(input_path, grid_size)
