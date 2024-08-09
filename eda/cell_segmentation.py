from cellpose import utils, models, io
import numpy as np
import time
import os
import sys
import cv2
from skimage.io import imread

IMAGE_EXTENSIONS = ['.jpg', '.bmp', '.png']
PATH = 'eda/dataset'
THRES_HOLD = 128


def get_bounding_boxes(masks):
    bounding_boxes = []
    unique_masks = np.unique(masks)
    height, width = img_shape[:2]

    for mask_value in unique_masks:
        if mask_value == 0:
            continue  # Bỏ qua background

        # Tạo một binary mask cho mỗi đối tượng
        binary_mask = (masks == mask_value).astype(np.uint8)

        # Tìm contour của đối tượng
        contours,_ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     for contour in contours:
    #         # Tìm bounding box từ contour
    #         x, y, w, h = cv2.boundingRect(contour)
    #         if w < THRES_HOLD:
    #             w = THRES_HOLD
    #         if h < THRES_HOLD:
    #             h = THRES_HOLD
    #         bounding_boxes.append((x, y, x + w, y + h))

    # return bounding_boxes
    for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Tính toán tâm của bounding box
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Tạo bounding box mới với kích thước THRES_HOLD x THRES_HOLD
            new_x = max(0, center_x - THRES_HOLD // 2)
            new_y = max(0, center_y - THRES_HOLD // 2)
            new_w = THRES_HOLD
            new_h = THRES_HOLD
            

            # Điều chỉnh nếu bounding box vượt quá biên của ảnh
            if new_x + new_w > width:
                new_x = width - new_w
            if new_y + new_h > height:
                new_y = height - new_h
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
    # for filename in files[:30]:
    #     img = io.imread(f'{PATH}/{filename}')
    #     print(f'Cropping cell on image: {filename}')

    #     masks, flows, styles, diams = model.eval(
    #         img, normalize=True, flow_threshold=0.3, diameter=None, channels=channels)

    #     name = filename.split('.')[0]
    #     bounding_boxes = get_bounding_boxes(masks)
        # In bounding boxes
        # for idx, bbox in enumerate(bounding_boxes):
        #     print(f"Bounding box: {bbox}")
        #     x1, y1, x2, y2 = bbox
        #     # Tạo một ảnh trống 128x128 với giá trị pixel là 0 (đen)
        #     cropped_image = np.zeros((THRES_HOLD, THRES_HOLD, 3), dtype=np.uint8)
            
        #     # Tính toán vị trí để paste ảnh gốc vào ảnh mới
        #     paste_x = max(0, (THRES_HOLD - (x2 - x1)) // 2)
        #     paste_y = max(0, (THRES_HOLD - (y2 - y1)) // 2)
            
        #     # Cắt và paste ảnh gốc vào ảnh mới
        #     # cropped_original = img[y1:y2, x1:x2]
        #     # cropped_image[paste_y:paste_y + cropped_original.shape[0], 
        #     #               paste_x:paste_x + cropped_original.shape[1]] = cropped_original
            
        #     output_dir = f'{PATH}/cropped_images'

            # os.makedirs(output_dir, exist_ok=True)
            # output_path = os.path.join(output_dir, f'{name}_{idx + 1}.png')
            # cv2.imwrite(output_path, cropped_image)
            # cropped_image = img[y1:y2, x1:x2]
            # output_dir = f'{PATH}/cropped_images'

            # os.makedirs(output_dir, exist_ok=True)

            # output_path = os.path.join(
            #     output_dir, f'{name}_{idx + 1}.png')
            # cv2.imwrite(output_path, cropped_image)
        # save results so you can load in gui
        # io.masks_flows_to_seg(img, masks, flows, diams, filename, chan)

        # save results as png
        # io.save_to_png(img, masks, flows, filename)
    # img = imread('eda/dataset/12.png')
    # masks, flows, styles, diams = model.eval(img, diameter=None)
    # num_segments = len(np.unique(masks)) - 1  # Subtract 1 to exclude the background
    
    # print(f"Number of segments: {num_segments}")  
    total_segments = []
    for filename in os.listdir(PATH):
        if filename.endswith('.tif') or filename.endswith('.png') or filename.endswith('.jpg'):
            # Load the image
            img_path = os.path.join(PATH, filename)
            img = imread(img_path)

            # Run Cellpose on the image
            masks, flows, styles, diams = model.eval(img, diameter=None)

            # Count the number of segments
            num_segments = len(np.unique(masks)) - 1  # Subtract 1 to exclude the background
            total_segments.append(num_segments)

    print(f"Total number of segments in the folder: {total_segments}")  
if __name__ == '__main__':
    main()
