import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image

gallery_dict = {}
query_dict = {}

gallery_dataset = {
    'whole': ImageFolder('dataset/market1501/bounding_box_test/whole', allow_empty=True),
    'head': ImageFolder('dataset/market1501/bounding_box_test/head', allow_empty=True),
    'upper_body': ImageFolder('dataset/market1501/bounding_box_test/upper_body', allow_empty=True),
    'lower_body': ImageFolder('dataset/market1501/bounding_box_test/lower_body', allow_empty=True),
}
query_dataset = {
    'whole': ImageFolder('dataset/market1501/query/whole', allow_empty=True),
    'head': ImageFolder('dataset/market1501/query/head', allow_empty=True),
    'upper_body': ImageFolder('dataset/market1501/query/upper_body', allow_empty=True),
    'lower_body': ImageFolder('dataset/market1501/query/lower_body', allow_empty=True),
}

for index, (file_path, _) in enumerate(gallery_dataset['whole'].samples):
    file_name = os.path.basename(file_path)
    gallery_dict[file_name] = { 'whole': index }

for part in ['head', 'upper_body', 'lower_body']:
    for index, (file_path, _) in enumerate(gallery_dataset[part].samples):
        file_name = os.path.basename(file_path)
        gallery_dict[file_name][part] = index

for index, (file_path, _) in enumerate(query_dataset['whole'].samples):
    file_name = os.path.basename(file_path)
    query_dict[file_name] = { 'whole': index }

for part in ['head', 'upper_body', 'lower_body']:
    for index, (file_path, _) in enumerate(query_dataset[part].samples):
        file_name = os.path.basename(file_path)
        query_dict[file_name][part] = index


sim_mats = {
    'whole': torch.from_numpy(np.load('saves/vectors/whole/sim.npy')),
    'head': torch.from_numpy(np.load('saves/vectors/head/sim.npy')),
    'upper_body': torch.from_numpy(np.load('saves/vectors/upper_body/sim.npy')),
    'lower_body': torch.from_numpy(np.load('saves/vectors/lower_body/sim.npy')),
}

harmonic_means = np.zeros((len(query_dict), len(gallery_dict)))

N = 0
for query_file_name, query_parts in tqdm(query_dict.items()):
    q_idx = query_parts['whole']
    for gallery_file_name, gallery_parts in gallery_dict.items():
        g_idx = gallery_parts['whole']
        count = 0.0
        sum = 0.0
        for part in ['whole', 'head', 'upper_body', 'lower_body']:
            if part in query_parts and part in gallery_parts:
                cos_sim = sim_mats[part][query_parts[part]][gallery_parts[part]]
                if abs(cos_sim) < 1e-9:
                    continue
                count += 1.0
                sum += 1.0 / cos_sim

            harmonic_means[q_idx, g_idx] = count / sum

    qimg = query_dataset['whole'].samples[q_idx][0]

    best_5 = np.argsort(harmonic_means[q_idx])[-5:][::-1]
    imgs = [gallery_dataset['whole'].samples[best][0] for best in best_5]

    to_display = [qimg] + imgs

    fig, axes = plt.subplots(1, len(to_display), figsize=(15, 3))

    # Plot each image
    for i, image_path in enumerate(to_display):
        # Load the image
        image = Image.open(image_path)
        
        # Display the image
        axes[i].imshow(image)
        axes[i].set_title(os.path.basename(to_display[i]))
        axes[i].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.savefig(f'./final/{q_idx}.png')

np.save('./final.npy', harmonic_means)