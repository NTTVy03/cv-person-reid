from models import model_loader
import faiss
from argparse import ArgumentParser
import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

BASE_DIR = 'dataset/market1501/pytorch/query'
NUM_CROPS = {
    'head': 3,
    'lower_body': 3,
    'upper_body': 3,
    'whole': 7
}


def load_db(path):
    return faiss.read_index(path)


def load_model(args):
    return model_loader.load_model_with_weight(args.model)


def load_images(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    images = {
        'head': [],
        'upper_body': [],
        'lower_body': [],
        'whole': []
    }

    for body_part in images.keys():
        body_dir = os.path.join(BASE_DIR, body_part)
        dirs = [dir for dir in os.listdir(body_dir) if dir.startswith(args.id)]
        for i in range(NUM_CROPS[body_part]):
            img_dir = os.path.join(os.path.join(body_dir, f'{args.id}_{i}'))
            if os.path.isdir(img_dir):
                img = Image.open(os.path.join(img_dir, f'{args.id}_{i}.jpg'))
                images[body_part].append(transform(img))
            else:
                images[body_part].append(None)

    return images


def extract_vector(images, args):
    model = load_model(args).to('cuda')

    result = {
        'head': [],
        'upper_body': [],
        'lower_body': [],
        'whole': []
    }

    with torch.no_grad():
        for body_part, crops in images.items():
            for crop in crops:
                if crop is not None:
                    crop = torch.unsqueeze(crop, 0)
                    vector = model(crop.to('cuda'))
                    vector = vector[0][0]
                else:
                    vector = torch.zeros(1024, device='cuda')
                result[body_part].append(vector)

    return result

def concat_vectors(feature_vectors):
    result = {
        'head': None,
        'upper_body': None,
        'lower_body': None,
        'whole': None
    }

    for body_part in result.keys():
        concat_vector = torch.cat(feature_vectors[body_part], dim=0)
        norm_vector = torch.nn.functional.normalize(concat_vector, dim=0)
        result[body_part] = norm_vector

    return result


def main():
    parser = ArgumentParser()

    parser.add_argument('--id')
    parser.add_argument('--model', '-m', default='swin_base')
    parser.add_argument('--db-path', '-dp')
    parser.add_argument('--device', '-d', choices=['cuda', 'cpu'], default='cuda')

    args = parser.parse_args()

    images = load_images(args)
    feature_vectors = extract_vector(images, args)
    to_query = concat_vectors(feature_vectors)
    for body_part in to_query.keys():
        index = load_db(os.path.join(args.db_path, body_part, 'db.index'))
        d, i = index.search(np.expand_dims(to_query[body_part].cpu().numpy(), axis=0), 10)
        print(body_part, i)


if __name__ == '__main__':
    main()


# from PIL import Image
# from models import model_loader
# from torchvision import transforms

# model = model_loader.load_model_with_weight('swin_base')
# img = Image.open('img_path')
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])
# img = transform(img)
# feature_vector = model(img.to('cuda'))