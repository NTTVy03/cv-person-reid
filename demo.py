import torch
import numpy as np

from argparse import ArgumentParser
from models import osnet
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image


query_img = Image.open('/home/ezio/code/PersonReID/dataset/market1501/query/lower_body/0019/0019_c3s3_075919_00.jpg')

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

query_tensor = transform(query_img)
query_batch = query_tensor.unsqueeze(0)

model = osnet.osnet_x0_25(num_classes=1503, pretrained=False)
model.load_state_dict(torch.load('saves/weights/lower_body_64x64_60.pt'))
model.cuda().eval()
with torch.no_grad():
    query_feature_map = model.featuremaps(query_batch.cuda())
    flattened = query_feature_map.view(query_feature_map.size(0), -1)
    normalized = F.normalize(flattened, p=2, dim=1)

gallery_mat = np.load('./saves/vectors/lower_body/gallery.npy')
gallery_tensor = torch.from_numpy(gallery_mat).cuda()

cosine_similarities = torch.mm(normalized, gallery_tensor.t())

best_indices = torch.topk(cosine_similarities, k=5, dim=1)

gallery_dataset = ImageFolder('/home/ezio/code/PersonReID/dataset/market1501/bounding_box_test/lower_body', transform=transform, allow_empty=True)

print(best_indices.values)
indices = best_indices.indices.tolist()[0]

for index in indices:
    print(index, gallery_dataset.targets[index])