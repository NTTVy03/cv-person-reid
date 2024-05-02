import os
from shutil import copy2
from argparse import ArgumentParser
import cv2


SRC_DATASET_PATH = '/home/ezio/Downloads/market1501'
DST_DATASET_PATH = 'dataset/market1501/pytorch'
SUB_DIRS = ['bounding_box_train', 'bounding_box_test', 'query']
BODY_PARTS = {
    '0': 'whole',
    '1': 'head',
    '2': 'upper_body',
    '3': 'lower_body'
}

# def split_image_vertically(num_split):
#     split_image()


def split_image(img, rows):
    height, _, _ = img.shape
    split_height = max(height // rows, 1)
    splitted_imgs = []
    for y in range(0, height, split_height):
        splitted_imgs.append(img[y:y+split_height, :])
    return splitted_imgs


def split_images(path, rows, start_idx=1):
    org_img_dirs = [dir for dir in os.listdir(path) if dir.endswith('_0')]

    for org_img_dir in org_img_dirs:
        org_img = cv2.imread(os.path.join(path, org_img_dir, org_img_dir + '.jpg'))
        splitted_images = split_image(org_img, rows)
        for i, splitted_image in enumerate(splitted_images):
            splitted_name = org_img_dir[:org_img_dir.rindex('_')+1] + str(i + start_idx)
            os.mkdir(os.path.join(path, splitted_name))
            cv2.imwrite(os.path.join(path, splitted_name, splitted_name + '.jpg'), splitted_image)


def init_convert():
    if not os.path.isdir(DST_DATASET_PATH):
        os.mkdir(DST_DATASET_PATH)

    for sub_dir in SUB_DIRS:
        if not os.path.isdir(os.path.join(DST_DATASET_PATH, sub_dir)):
            os.mkdir(os.path.join(DST_DATASET_PATH, sub_dir))


def convert_sub_directory(sub_dir):
    if not (sub_dir in SUB_DIRS):
        raise Exception(f'{sub_dir} is not a valid sub directory of Market1501.')
    
    src_sub_dir_path = os.path.join(SRC_DATASET_PATH, sub_dir)
    dst_sub_dir_path = os.path.join(DST_DATASET_PATH, sub_dir)

    for body_part_sub_dir in BODY_PARTS.values():
        if not os.path.isdir(os.path.join(dst_sub_dir_path, body_part_sub_dir)):
            os.mkdir(os.path.join(dst_sub_dir_path, body_part_sub_dir))

    for src_image_path in os.listdir(src_sub_dir_path):
        if not src_image_path.endswith('.jpg'):
            continue

        image_name = os.path.basename(src_image_path).split('.')[0]
        separator_pos = image_name.find('_')
        body_part_key, org_name = image_name[:separator_pos], image_name[separator_pos+1:] + '_0'

        dst_dir_path = os.path.join(dst_sub_dir_path, BODY_PARTS[body_part_key], org_name)
        if not os.path.isdir(dst_dir_path):
            os.mkdir(dst_dir_path)

        copy2(os.path.join(src_sub_dir_path, src_image_path), os.path.join(dst_dir_path, org_name + '.jpg'))


def main():


    init_convert()
    # convert_sub_directory('query')
    convert_sub_directory('bounding_box_test')

    # split_images('dataset/market1501/pytorch/query/head', 2)
    # split_images('dataset/market1501/pytorch/query/lower_body', 2)
    # split_images('dataset/market1501/pytorch/query/upper_body', 2)
    # split_images('dataset/market1501/pytorch/query/whole', 2)
    # split_images('dataset/market1501/pytorch/query/whole', 4, 3)

    split_images('dataset/market1501/pytorch/bounding_box_test/head', 2)
    split_images('dataset/market1501/pytorch/bounding_box_test/lower_body', 2)
    split_images('dataset/market1501/pytorch/bounding_box_test/upper_body', 2)
    split_images('dataset/market1501/pytorch/bounding_box_test/whole', 2)
    split_images('dataset/market1501/pytorch/bounding_box_test/whole', 4, 3)


if __name__ == '__main__':
    main()