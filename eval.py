from models import model_loader

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import faiss
from argparse import ArgumentParser
import json
import os


def model_predict(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = ImageFolder(root='dataset/market1501/pytorch/bounding_box_test/head', transform=transform)

    batch_size = 32
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    label_mapping = {idx: cls for cls, idx in data_loader.dataset.class_to_idx.items()}

    return label_mapping

    if args.output_label_path:
        with open(args.output_label_path, 'w') as f:
            f.write(json.dumps(label_mapping))

    model = model_loader.load_model_with_weight('swin_base')
    model.eval()
    model.to(args.device)


    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            x, y = data
            x = x.to(args.device)
            pred = model(x)
            for i in range(pred[0].shape[0]):
                torch.save(pred[0][i], os.path.join(args.output_dir, label_mapping[y[i].item()] + '.pt'))

    return label_mapping


def generate_database(path, label_mapping, args):
    index = faiss.IndexFlatIP(1024 * args.size)


    distinct_ids = {}
    for key, value in label_mapping.items():
        id = value[:value.rindex('_')]
        if id in distinct_ids.keys():
            distinct_ids[id].append(value)
        else:
            distinct_ids[id] = [value]

    for id, vector_ids in distinct_ids.items():
        vectors_list = []
        for i in range(args.size):
            vector_path = os.path.join(path, f'{id}_{i}.pt')
            if os.path.exists(vector_path):
                vector = torch.load(vector_path)
            else:
                vector = torch.zeros(1024, device=args.device)
            vectors_list.append(vector)
        concat_vector = torch.cat(vectors_list, dim=0)
        concat_vector = torch.nn.functional.normalize(concat_vector, dim=0)
        index.add(concat_vector)
 
    faiss.write_index(index, 'output.index')
    # index = faiss.IndexFlatIP(1024)


def main():
    parser = ArgumentParser()
        
    parser.add_argument('--body-part', '-bp')
    parser.add_argument('--device', '-d', choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--output-label-path', '-ol')
    parser.add_argument('--output-dir', '-o')
    parser.add_argument('--size', '-s', type=int, default=3)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = './output'
    
    label_mapping = model_predict(args)

    generate_database('./output', label_mapping, args)


if __name__ == '__main__':
    main()