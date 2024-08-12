from cellpose import utils, models, io
import numpy as np
import time
import os
import sys
import cv2
import shutil
import pandas as pd
from tqdm import tqdm

IMAGE_EXTENSIONS = ['.jpg', '.bmp', '.png']
PATH = 'eda/dataset'
THRES_HOLD = 128


def get_bounding_boxes(masks):
    bounding_boxes = []
    unique_masks = np.unique(masks)

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
            if w < THRES_HOLD:
                w = THRES_HOLD
            if h < THRES_HOLD:
                h = THRES_HOLD
            bounding_boxes.append((x, y, x + w, y + h))

    return bounding_boxes


def load_dataset(path):
    lst_path = [file for file in os.listdir(path) if file != '.DS_Store']
    lst_path = sorted(lst_path, key=lambda x: int(x.split('.')[0]))
    return lst_path

def main():
    # RUN CELLPOSE
    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=False, model_type='nuclei')

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
    for filename in files[:2]:
        img = io.imread(f'{PATH}/{filename}')
        print(f'Cropping cell on image: {filename}')

        masks, flows, styles, diams = model.eval(
            img, normalize=True, flow_threshold=0.3, diameter=None, channels=channels)

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

def split_dataset(root_dir, target_dir, val_ratio=0.15, test_ratio=0.15):
    # Creating Split Folders
    os.makedirs(target_dir + '/train/', exist_ok=True)
    os.makedirs(target_dir + '/val/', exist_ok=True)
    os.makedirs(target_dir + '/test/', exist_ok=True)

    # Folder to copy images from
    src = os.path.join(root_dir)

    # Spliting the Files in the Given ratio
    all_filenames = os.listdir(src)
    np.random.shuffle(all_filenames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(all_filenames), [int(len(
        all_filenames) * (1 - (val_ratio + test_ratio))), int(len(all_filenames) * (1 - test_ratio))])

    train_FileNames = [src + '/' +
                       name for name in train_FileNames.tolist()]
    val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

    # Printing the Split Details
    print('Total images: ', len(all_filenames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        os.makedirs(target_dir + '/train/', exist_ok=True)
        shutil.copy(name, f'{target_dir}/train')

    for name in val_FileNames:
        os.makedirs(target_dir + '/val/', exist_ok=True)
        shutil.copy(name, f'{target_dir}/val')

    for name in test_FileNames:
        os.makedirs(target_dir + '/test/', exist_ok=True)
        shutil.copy(name, f'{target_dir}/test')

def analysis_cell_distribution(lst_path):
    # RUN CELLPOSE
    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=False, model_type='cyto3', )

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

    print('== Start counting cell for each image')
    lst_cell_analysis = []
    print(lst_path)
    for index, filename in tqdm(enumerate(lst_path)):
        print(f'Processing on epoch: {index}')
        if filename.endswith('.tif') or filename.endswith('.png') or filename.endswith('.jpg'):
            # Load the image
            img_path = os.path.join(PATH, filename)
            img = imread(img_path)

            # Run Cellpose on the image
            masks, flows, styles, diams = model.eval(img, diameter=None)

            # Count the number of segments
            # Subtract 1 to exclude the background
            num_segments = len(np.unique(masks)) - 1
            lst_cell_analysis.append(num_segments)
            print(f"Number of segments in {filename}: {num_segments}")

    # print(f"Total number of segments in the folder: {len(lst_cell_analysis)}")
    return lst_cell_analysis


if __name__ == '__main__':
    lst_path = sorted(load_dataset(
        PATH), key=lambda x: int(x.split('.')[0]))
    print(lst_path)
    # lst_path = load_dataset(PATH)
    # for item in lst_path:
    #     try:
    #         number = int(item.split('.')[0])
    #         print(f'number : {number}')
    #     except:
    #         print(f'error on item: {item}')
    # split_dataset(root_dir='eda/smear_test/converted',
    #   target_dir='src/dataset/target')
