import os

from shutil import copy2
from argparse import ArgumentParser


ID_BODYPART = {
    0: 'whole',
    1: 'head',
    2: 'upper_body',
    3: 'lower_body'
}
CLASSES = ['-1'] + ['{:04d}'.format(i) for i in range(1501 + 1)]


def main():
    parser = ArgumentParser()
    parser.add_argument('--source', '-s')
    parser.add_argument('--destination', '-d')
    args = parser.parse_args()

    if not os.path.isdir(args.destination):
        os.mkdir(args.destination)
    for body_part_name in ID_BODYPART.values():
        body_part_dir = os.path.join(args.destination, body_part_name)
        if not os.path.isdir(body_part_dir):
            os.mkdir(body_part_dir)
        for class_name in CLASSES:
            os.mkdir(os.path.join(body_part_dir, class_name))

    for img_path in os.listdir(args.source):
        img_name = img_path.split('.')[0]
        sep_pos = img_name.find('_')
        body_part_id, img_id = img_name[:sep_pos], img_name[sep_pos+1:]
        person_id = img_id[:img_id.find('_')]
        body_part = ID_BODYPART[int(body_part_id)]
        dst_dir = os.path.join(args.destination, body_part, person_id)
        copy2(os.path.join(args.source, img_path), os.path.join(dst_dir, img_id + '.jpg'))


if __name__ == '__main__':
    main()
