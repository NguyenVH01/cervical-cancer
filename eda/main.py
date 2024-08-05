import os
import pandas as pd
from PIL import Image

CSV_PATH = 'eda/Data_clean_v1.csv'
IMAGE_PATH = 'eda/smear_test'
IMAGE_COL = 'TÊN FILE'
IMAGE_PATH_COL = 'ĐƯỜNG DẪN FILE'
IMAGE_EXTENSIONS = ['.jpg', '.bmp', '.png']
DATASET_PATH = 'src/dataset'


def read_data_csv(path):
    """Đọc dữ liệu từ file CSV."""
    return pd.read_csv(path, sep=';')

def read_image_info(path):
    """Đọc danh sách các file hình ảnh từ thư mục."""
    return [file for file in os.listdir(path) if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS]

def rename_images(image_path_files, image_list, image_col=IMAGE_COL):
    """Đổi tên các file hình ảnh theo thứ tự."""
    new_paths = []
    for idx, original_name in enumerate(image_path_files[image_col], start=1):
        new_name = f"{idx}.png"
        new_paths.append(new_name)

        if original_name in image_list:
            os.rename(os.path.join(IMAGE_PATH, original_name),
                      os.path.join(IMAGE_PATH, new_name))

    image_path_files[image_col] = new_paths

    image_path_files.drop(IMAGE_PATH_COL, axis=1, inplace=True)

    image_path_files.to_csv('eda/clean_v2.csv', index=False)
    print(image_path_files)

def convert_images_to_png():
    """Chuyển đổi các tệp hình ảnh từ các định dạng khác sang PNG."""
    for ids, file_name in enumerate(os.listdir(IMAGE_PATH)):
        # Lấy đường dẫn đầy đủ của tệp hình ảnh
        file_path = os.path.join(IMAGE_PATH, file_name)

        # Bỏ qua nếu không phải là tệp hình ảnh
        if not os.path.isfile(file_path):
            continue

        if os.path.splitext(file_path)[1].lower() in IMAGE_EXTENSIONS:
            # Mở tệp hình ảnh
            with Image.open(file_path) as img:
                os.makedirs('eda/smear_test/converted', exist_ok=True)
                # Chuyển đổi tệp hình ảnh sang PNG
                output_file_name = str(ids + 1) + '.png'
                output_file_path = os.path.join(
                    IMAGE_PATH + '/converted', output_file_name)
                img.save(output_file_path, 'PNG')
                print(f"Converted {file_name} to {output_file_name}")

# def split_dataset(is_target = False):


def main():
    image_list = read_image_info(IMAGE_PATH)
    dataframe = read_data_csv(CSV_PATH)

    images_not_included = dataframe[~dataframe[IMAGE_COL].isin(image_list)]

    print(f'Size of images before clean: {len(dataframe[IMAGE_COL])}')
    print(f'Size of images not having in CSV: {len(images_not_included)}')
    print(f'Size of images after clean: {
          len(dataframe) - len(images_not_included)}')

    dataframe = dataframe[dataframe[IMAGE_COL].isin(image_list)]
    rename_images(dataframe, image_list)


if __name__ == '__main__':
    # main()
    convert_images_to_png()
