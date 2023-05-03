import cv2
import os

def convert_bmp_to_png(src_path, dest_folder):
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.lower().endswith(".bmp"):
                # Read the BMP image
                bmp_path = os.path.join(root, file)
                img = cv2.imread(bmp_path)

                # Check if the image was read successfully
                if img is None:
                    print(f"Error: Unable to read image at {bmp_path}")
                    continue

                # Create the destination folder if it doesn't exist
                os.makedirs(dest_folder, exist_ok=True)

                # Get the image filename without extension
                base_filename = os.path.splitext(file)[0]

                # Set the output path for the PNG image
                dest_path = os.path.join(dest_folder, f"{base_filename}.png")

                # Save the image as a PNG
                cv2.imwrite(dest_path, img)
                print(f"Image converted and saved at {dest_path}")

# Example usage:
src_path = "/nas/ai_image/sync_image/baidu_pan_download/mask_samples/"
dest_folder = "/nas/ai_image/sync_image/baidu_pan_download/mask_samples_png/"
convert_bmp_to_png(src_path, dest_folder)
