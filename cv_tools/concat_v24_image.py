import os
import cv2
import numpy as np
from math import ceil
from glob import glob

def concat_images(input_path, output_path=None, batch_size=24, target_size=(1696,920),  fill_color=[113, 119, 52]):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(input_path),f"{os.path.dirname(input_path)}") 
    os.makedirs(output_path, exist_ok=True)

    image_paths = glob(os.path.join(input_path, '*.png'))  # Adjust the pattern as needed
    print(f"contains:{len(image_paths)}")
    batch_count = ceil(len(image_paths) / batch_size)
    cell_height = target_size[1] // 4  # Divide the target height by the number of rows
    cell_width = target_size[0] // 6  # Divide the target width by the number of columns
    print(f"batch_count:{batch_count}")
    for i in range(batch_count):
        print(f"processing batch {i}")
        batch_paths = image_paths[i*batch_size:(i+1)*batch_size]
        while len(batch_paths) < batch_size:  # If the batch is smaller than 24, fill it with empty images
            empty_img = np.full((cell_height, cell_width, 3), fill_color, dtype=np.uint8)
            batch_paths.append(empty_img)

        images = []
        for path in batch_paths:
            if isinstance(path, str):
                img = cv2.imread(path)
                h, w, _ = img.shape
                cell = np.full((cell_height, cell_width, 3), fill_color, dtype=np.uint8)

                # Scenario 1: both width and height are smaller
                if h < cell_height and w < cell_width:
                    y = (cell_height - h) // 2
                    x = (cell_width - w) // 2
                    cell[y:y+h, x:x+w] = img

                # Scenario 2: width is greater, height is smaller
                elif h < cell_height and w >= cell_width:
                    x = (w - cell_width) // 2
                    x_end = x + cell_width if w % 2 == 0 else x + cell_width + 1
                    y = (cell_height - h) // 2
                    cell[y:y+h, :] = img[:, x:x_end]

                # Scenario 3: height is greater, width is smaller
                elif h >= cell_height and w < cell_width:
                    y = (h - cell_height) // 2
                    y_end = y + cell_height if h % 2 == 0 else y + cell_height + 1
                    x = (cell_width - w) // 2
                    cell[:, x:x+w] = img[y:y_end, :]

                # Scenario 4: both width and height are greater
                else:  # h >= cell_height and w >= cell_width
                    y = (h - cell_height) // 2
                    y_end = y + cell_height if h % 2 == 0 else y + cell_height + 1
                    x = (w - cell_width) // 2
                    x_end = x + cell_width if w % 2 == 0 else x + cell_width + 1
                    cell = img[y:y_end, x:x_end]

                img = cell
            else:
                img = path  # It's an empty image
            images.append(img)

            #images.append(img)

        # Before stacking, check the shapes of the images
        for j in range(0, batch_size, 6):
            shapes = [img.shape for img in images[j:j+6]]
            #print(f"Shapes of images in row {i//6}: {shapes}")

        # Concatenate the images into a grid
        rows = [np.hstack(images[i:i+6]) for i in range(0, batch_size, 6)]
        final_img = np.vstack(rows) 

        # Do the same for vstack
        # print("Shapes of rows:")
        # print([row.shape for row in rows])

        # Resize final image to the target size
        final_img = cv2.resize(final_img, target_size, interpolation=cv2.INTER_CUBIC)
        
        save_path = os.path.join(output_path, f"{i}.png")
        print(f"saving batch {i} to {save_path}")
        cv2.imwrite(save_path, final_img)

def walk_subdirectories(input_path, output_path):
    for dirpath, dirnames, filenames in os.walk(input_path):
        has_image = any(filename.endswith(('.png', '.jpg', '.jpeg')) for filename in filenames)
        if has_image:
            print(dirpath)
            relative_dir = os.path.relpath(dirpath, input_path)
            output_dir = os.path.join(output_path, relative_dir)
            print(f"input: {dirpath}")
            print(f"output: {output_dir}")
            concat_images(dirpath, output_dir)



if __name__ == "__main__":
    #concat_images("/nas/ai_image/sync_image/baidu_pan_download/灰枣_以前切图/二次样本-变形-已复检14929/二次样本-变形-已复检14929/一级变形-已复检-547/", "/opt/images/broken/二次样本一级变形-已复检-547/")
    walk_subdirectories("/opt/images/UAE/khalash/通用样本/21-11-07%2010-14-44/slice/winkle/","/opt/images/UAE/khalash/通用样本/")