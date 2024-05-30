import os

import torch

from models import osnet
from argparse import ArgumentParser
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


NUM_CLASSES = 1503
INPUT_SHAPES = {
    'whole': (128, 64),
    'head': (32, 64),
    'upper_body': (64, 64),
    'lower_body': (64, 64)
}


def train():
    parser = ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=60)
    parser.add_argument('--part', choices=['whole', 'head', 'upper_body', 'lower_body'])
    args = parser.parse_args()

    input_shape = INPUT_SHAPES[args.part]

    transform = transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = ImageFolder(args.dataset, transform=transform, allow_empty=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    model = osnet.osnet_x0_25(pretrained=False, num_classes=NUM_CLASSES).cuda()
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if not os.path.isdir('./runs'):
        os.mkdir('./runs')
    writer = SummaryWriter()

    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')

    for epoch in range(args.num_epochs):
        train_loss = 0.0
        for inputs, targets in tqdm(data_loader, desc=f'Epoch {epoch + 1}'):
            optimizer.zero_grad()
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        scheduler.step()
        train_loss /= len(dataset)

        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'./checkpoints/{args.part}_{input_shape[0]}x{input_shape[1]}_{epoch + 1}.pt')


if __name__ == '__main__':
    train()