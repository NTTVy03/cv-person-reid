import numpy as np

from torchvision.datasets import ImageFolder


def top_k_acc(scores, query, gallery, k=5):
    scores_cp = scores.copy()
    count = 0
    top_k_indices = np.argsort(scores_cp, axis=1)[:, -k:][:, ::-1]
    for query_id in range(top_k_indices.shape[0]):
        k_indices  = [gallery.targets[index] for index in top_k_indices[query_id]]
        if query.targets[query_id] in k_indices:
            count += 1
    return count / top_k_indices.shape[0]


def get_class_preds(scores, query, gallery):
    y_scores = np.zeros((len(query), len(gallery.classes)))
    counts = np.array([1e-9] * len(gallery.classes))
    for g in range(scores.shape[1]):
        gc = gallery.targets[g]
        y_scores[:, gc] += scores[:, g]
        counts[gc] += 1.0

    y_scores = y_scores / counts
    row_sums = y_scores.sum(axis=1)
    y_scores = y_scores / row_sums[:, np.newaxis]

    return y_scores


def avg_prec(y_true, y_scores):
    indices = np.argsort(-y_scores)
    y_true = y_true[indices]
    y_scores = y_scores[indices]    
    
    cum_true_positives = np.cumsum(y_true)
    total_positives = np.sum(y_true)

    if total_positives == 0:
        return 0.0

    precision = cum_true_positives / (np.arange(len(y_true)) + 1)
    recall = cum_true_positives / total_positives
    
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])

    return ap


def mean_avg_prec(y_true, y_scores):
    num_classes = y_true.shape[1]
    average_precisions = []
    
    for i in range(num_classes):
        ap = avg_prec(y_true[:, i], y_scores[:, i])
        average_precisions.append(ap)
    
    return np.mean(average_precisions)


# scores = np.load('saves/scores.npy')
scores = np.load('./final.npy')


gallery_dataset = ImageFolder('dataset/market1501/bounding_box_test/whole', allow_empty=True)
query_dataset = ImageFolder('dataset/market1501/query/whole', allow_empty=True)

for k in [1, 3, 5, 10]:
    acc = top_k_acc(scores, query_dataset, gallery_dataset, k)
    print(f'Top {k} accuracy: {acc}')

preds = get_class_preds(scores, query_dataset, gallery_dataset)

targets = np.array(query_dataset.targets)
targets_one_hot = np.zeros((len(query_dataset), len(gallery_dataset.classes)))
targets_one_hot[np.arange(len(query_dataset)), targets] = 1

mAP = mean_avg_prec(targets_one_hot, preds)

print(f'mAP: {mAP}')