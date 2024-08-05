from cellpose import utils, models, io
import numpy as np
import time
import os
import sys
import cv2

IMAGE_EXTENSIONS = ['.jpg', '.bmp', '.png']
PATH = 'eda/dataset'
THRES_HOLD = 32

def get_bounding_boxes(masks):
    bounding_boxes = []
    unique_masks = np.unique(masks)
    height, width = masks.shape

    for mask_value in unique_masks:
        if mask_value == 0:
            continue  # Bỏ qua background

        # Tạo một binary mask cho mỗi đối tượng
        binary_mask = (masks == mask_value).astype(np.uint8)

        # Tìm contour của đối tượng
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Tìm bounding box từ contour
            x, y, w, h = cv2.boundingRect(contour)
            # Xử lý vùng biên của ảnh
            x_start = max(x, 0)
            y_start = max(y, 0)
            x_end = min(x + w, THRES_HOLD)
            y_end = min(y + h, THRES_HOLD)

            bounding_boxes.append((x_start, y_start, x_end, y_end))

    return bounding_boxes

def load_dataset(path):
    return [file for file in os.listdir(path)]

def main():
    # RUN CELLPOSE
    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=False, model_type='cyto')

    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    # channels = [0,0]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    channels = [0, 0]  # IF YOU HAVE GRAYSCALE
    # channels = [2, 3]  # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # or if you have different types of channels in each image
    # channels = [[0, 0], [2, 3], [0, 0]]

    files = load_dataset(PATH)

    # or in a loop
    for filename in files:
        img = io.imread(f'{PATH}/{filename}')
        print(f'Cropping cell on image: {filename}')

        masks, flows, styles, diams = model.eval(
            img, normalize=True, diameter=50, channels=channels)

        name = filename.split('.')[0]
        bounding_boxes = get_bounding_boxes(masks)
        # In bounding boxes
        for idx, bbox in enumerate(bounding_boxes):
            print(f"Bounding box: {bbox}")
            x1, y1, x2, y2 = bbox
            cropped_image = img[y1:y2, x1:x2]
            output_dir = f'{PATH}/cropped_images'

            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(
                output_dir, f'{name}_{idx + 1}.png')
            cv2.imwrite(output_path, cropped_image)
        # save results so you can load in gui
        # io.masks_flows_to_seg(img, masks, flows, diams, filename, chan)

        # save results as png
        # io.save_to_png(img, masks, flows, filename)


if __name__ == '__main__':
    main()
