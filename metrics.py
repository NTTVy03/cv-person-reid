import numpy as np

from torchvision.datasets import ImageFolder


def top_k_acc(scores, query, gallery, k=5):
    count = 0
    top_k_indices = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    for query_id in range(top_k_indices.shape[0]):
        k_indices  = [gallery.targets[index] for index in top_k_indices[query_id]]
        if query.targets[query_id] in k_indices:
            count += 1
    return count / top_k_indices.shape[0]


scores = np.load('saves/scores.npy')

gallery_dataset = ImageFolder('dataset/market1501/bounding_box_test/whole', allow_empty=True)
query_dataset = ImageFolder('dataset/market1501/query/whole', allow_empty=True)

for k in [1, 3, 5, 10]:
    acc = top_k_acc(scores, query_dataset, gallery_dataset, k)
    print(f'Top {k} accuracy: {acc}')