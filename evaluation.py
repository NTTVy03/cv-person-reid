import os

import torch
import numpy as np

from argparse import ArgumentParser
from models import osnet
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


def evalutation():
    parser = ArgumentParser()
    parser.add_argument('--gallery', '-g')
    parser.add_argument('--query', '-q')
    parser.add_argument('--weight')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    weight_file_name = os.path.basename(args.weight)
    input_size = weight_file_name.split('.')[0].split('_')[-2]
    height, width = input_size.split('x')

    model = osnet.osnet_x0_25(num_classes=1503, pretrained=False)
    model.load_state_dict(torch.load(args.weight))
    model.eval().cuda()

    transform = transforms.Compose([
        transforms.Resize((int(height), int(width))),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    gallery_dataset = ImageFolder(args.gallery, transform=transform, allow_empty=True)
    query_dataset = ImageFolder(args.query, transform=transform, allow_empty=True)

    gallery_dataloader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    with torch.no_grad():
        with tqdm(total=len(gallery_dataset), desc='Gallery') as pbar:
            for batch_idx, (inputs, _) in enumerate(gallery_dataloader):
                n = inputs.size(0)
                inputs = inputs.cuda()
                feature_maps = model.featuremaps(inputs)
                flattened = feature_maps.view(feature_maps.size(0), -1)
                normalized = F.normalize(flattened, p=2, dim=1)
                if batch_idx == 0:
                    gallery_features = torch.zeros(len(gallery_dataset), normalized.size(1))
                gallery_features[batch_idx*args.batch_size:batch_idx*args.batch_size+n] = normalized
                pbar.update(n)

    with open('./vectors/gallery.npy', 'wb') as f:
        np.save(f, gallery_features.cpu().numpy())

    with torch.no_grad():
        with tqdm(total=len(query_dataset), desc='Query') as pbar:
            for batch_idx, (inputs, _) in enumerate(query_dataloader):
                n = inputs.size(0)
                inputs = inputs.cuda()
                feature_maps = model.featuremaps(inputs)
                flattened = feature_maps.view(feature_maps.size(0), -1)
                normalized = F.normalize(flattened, p=2, dim=1)
                if batch_idx == 0:
                    query_features = torch.zeros(len(query_dataset), normalized.size(1))
                query_features[batch_idx*args.batch_size:batch_idx*args.batch_size+n] = normalized
                pbar.update(n)

    with open('./vectors/query.npy', 'wb') as f:
        np.save(f, query_features.cpu().numpy())

    cosine_similarities = torch.mm(query_features, gallery_features.t())
    with open('./vectors/sim.npy', 'wb') as f:
        np.save(f, cosine_similarities.cpu().numpy())


if __name__ == '__main__':
    evalutation()