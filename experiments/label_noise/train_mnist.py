import argparse
import os
import json

import torch

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--runid', type=str, default='0_0_0', required=False)
args = parser.parse_args()

device = torch.device('cuda:0')
 
image_pipline=[]
label_pipline=[IntDecoder(), ToTensor(), Squeeze()]
build_loader = lambda path, **kwargs: Loader(
    path,
    num_workers=8,
    pipelines={
        'image': image_pipline,
        'label': label_pipline,
    },
    **kwargs,
)

train_loader = build_loader(
    path=f'/tmp/{os.environ["USER"]}/datasets/mnist_train__{args.runid}.beton',
    batch_size=64,
    order=OrderOption.RANDOM,
)

test_loaders = [
    build_loader(
        path=f'/tmp/{os.environ["USER"]}/datasets/mnist_test_{i}__{args.runid}.beton',
        batch_size=1000,
        order=OrderOption.SEQUENTIAL,
        drop_last=False,
    )
    for i in range(2)
]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
    print(f'Epoch: {epoch}, loss={loss.item()}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    correct_mod = [0,0]
    total_mod = [0,0]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            for g in [0,1]:
                correct_mod[g] += (pred.eq(target.view_as(pred))*(target.view_as(pred)%2==g)).sum().item()
                total_mod[g] += (target.view_as(pred)%2==g).sum().item()

    acc = correct/total
    acc_mod = [correct/total for correct,total in zip(correct_mod,total_mod)]
    print(f'acc={acc:.3f} ({correct}/{total})')
    return {
        'acc': acc,
        'acc_even': acc_mod[0],
        'acc_odd': acc_mod[1],
        'total': total,
        'total_even': total_mod[0],
        'total_odd': total_mod[1],
    }

# Training settings
epochs = 200

model = Net().to(device)
print('Number of parameters:', sum(p.numel() for p in model.parameters()))
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

results = []
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    results.append({
        'epoch': epoch,
        'group_a': test(model, device, test_loaders[0]),
        'group_b': test(model, device, test_loaders[1]),
        'training_set': test(model, device, train_loader),
    })

json.dump(
    obj=results,
    fp=open(f'{args.runid}.json','w'),
)


