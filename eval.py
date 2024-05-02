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

    dataset = ImageFolder(root=args.path, transform=transform)

    batch_size = 32
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    label_mapping = {idx: cls for cls, idx in data_loader.dataset.class_to_idx.items()}

    with open(os.path.join(args.output_path, 'label_mapping.json'), 'w') as f:
        f.write(json.dumps(label_mapping))

    model = model_loader.load_model_with_weight('swin_base')
    model.eval()
    model.to(args.device)

    dump_dir_path = os.path.join(args.output_path, 'dump')

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            x, y = data
            x = x.to(args.device)
            pred = model(x)
            for i in range(pred[0].shape[0]):
                torch.save(pred[0][i], os.path.join(dump_dir_path, label_mapping[y[i].item()] + '.pt'))

    return label_mapping


def generate_database(label_mapping, args):
    index = faiss.IndexFlatIP(1024 * args.size)

    dump_dir_path = os.path.join(args.output_path, 'dump')

    distinct_ids = {}
    for key, value in label_mapping.items():
        id = value[:value.rindex('_')]
        if id in distinct_ids.keys():
            distinct_ids[id].append(value)
        else:
            distinct_ids[id] = [value]

    db_idx_mapping = []
    for id, vector_ids in distinct_ids.items():
        db_idx_mapping.append(id)
        vectors_list = []
        for i in range(args.size):
            vector_path = os.path.join(dump_dir_path, f'{id}_{i}.pt')
            if os.path.exists(vector_path):
                vector = torch.load(vector_path)
            else:
                vector = torch.zeros(1024, device=args.device)
            vectors_list.append(vector)
        concat_vector = torch.cat(vectors_list, dim=0)
        concat_vector = torch.nn.functional.normalize(concat_vector, dim=0)
        index.add(concat_vector.reshape(1, -1).to('cpu'))
 
    faiss.write_index(index, os.path.join(args.output_path, 'db.index'))
    db_idx_mapping = {idx: id for idx, id in enumerate(db_idx_mapping)}
    with open(os.path.join(args.output_path, 'db_idx_mapping.json'), 'w') as f:
        f.write(json.dumps(db_idx_mapping))

    return db_idx_mapping


def main():
    parser = ArgumentParser()
        
    parser.add_argument('--path', '-p')
    parser.add_argument('--output-path', '-o')
    parser.add_argument('--device', '-d', choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--size', '-s', type=int, default=3)

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = './output'
    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)
    if not os.path.isdir(os.path.join(args.output_path, 'dump')):
        os.mkdir(os.path.join(args.output_path, 'dump'))
    

    label_mapping = model_predict(args)

    generate_database(label_mapping, args)


if __name__ == '__main__':
    main()