import argparse

import numpy as np
import pandas as pd

import torch
import torchvision

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField, TorchTensorField
import json

import dataset_utils

parser = argparse.ArgumentParser()
parser.add_argument('--runid', type=str, default='0_0_0', required=False)
parser.add_argument('--p', type=float, default=0.0, required=False)
args = parser.parse_args()

raw_datasets = dataset_utils.get_torchvision_dataset('MNIST')
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
torchvision_datasets = {k:dataset_utils.DatasetTransform(v,transform) for k,v in raw_datasets.items()}

rng = np.random.default_rng()
# 0.7/0.3 is good
imbalance = 0.8
majority_noise = 0.2
print(f'imbalance={imbalance}, noise={majority_noise}')

build_confusion_matrix = lambda g,noise_p,output_dim=10: np.array([
    [
        (i==j)*(1-noise_p*(i%2==g))
        + (i%2==g)*(i!=j)*((j-i)%output_dim==2)*noise_p
        for j in range(output_dim)
    ]
    for i in range(output_dim)
])
print(build_confusion_matrix(0,majority_noise))
print(build_confusion_matrix(1,majority_noise))

datasets = {
    dataset_type: [
        dataset_utils.DatasetLabelNoise(
            dataset=ds,
            confusion_matrix=build_confusion_matrix(g=i,noise_p=majority_noise),
            random_state=rng,
        )
        for i,ds in enumerate(dataset_utils.split_dataset(
            dataset=torchvision_datasets[dataset_type],
            class_imbalance=imbalance,
            random_state=rng,
        ))
    ]
    for dataset_type in ['train','test']
}

write_dataset = lambda dataset, path: DatasetWriter(
    path, 
    {
        # 'image': RGBImageField(),
        'image': TorchTensorField(dtype=torch.float32, shape=(1,28,28)),
        'label': IntField()
    },
).from_indexed_dataset(dataset)

print(f'mix_p={args.p}')
trainset = dataset_utils.mix_datasets(
    datasets=datasets['train'],
    p=[1-args.p,args.p],
    random_state=rng,
)
write_dataset(
    dataset=trainset,
    path=f'{dataset_utils.output_path}/mnist_train__{args.runid}.beton',
)
test_y = pd.Series([y for x,y in trainset])
print(test_y.value_counts().pipe(lambda s: s/s.sum()))
print(test_y.map(lambda x: x%2).value_counts().pipe(lambda s: s/s.sum()))

for i,ds in enumerate(datasets['test']):
    write_dataset(
        dataset=ds,
        path=f'{dataset_utils.output_path}/mnist_test_{i}__{args.runid}.beton',
    )
    