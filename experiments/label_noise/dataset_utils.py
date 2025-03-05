import numpy as np
import torch
import torchvision
import torch.utils.data as torchdata
import os


# torchvision_datasets_path = '/datasets'
torchvision_datasets_path = f'/tmp/{os.environ["USER"]}/torchvision_datasets'
output_path = f'/tmp/{os.environ["USER"]}/datasets'
assert os.path.exists(output_path), f'Output path {output_path} does not exist'

class DatasetTransform(torchdata.Dataset):
    def __init__(self, dataset, transform=None):
        self.ds = dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample,target = self.ds[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample,target


class DatasetLabelNoise(torchdata.Dataset):
    def __init__(self, dataset, confusion_matrix, random_state):
        self.rng = np.random.default_rng(random_state)
        self.ds = dataset
        self.orig_y = np.array([y for x,y in dataset], dtype=int)
        self.confusion = np.array(confusion_matrix)
        if not np.allclose(np.array(confusion_matrix).sum(axis=1),1):
            raise AssertionError(f'Invalid confusion matrix. row sums: {np.array(confusion_matrix).sum(axis=1)}')
        self.y = np.zeros(len(self.orig_y), dtype=int)-1
        output_dim = len(confusion_matrix)
        for y in range(output_dim):
            self.y[self.orig_y==y] = self.rng.choice(
                a=output_dim,
                p=confusion_matrix[y],
                size=(self.orig_y==y).sum(),
            )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample,target = self.ds[idx]
        noisy_target = self.y[idx]
        return sample,noisy_target


def shuffle_dataset(dataset, random_state):
    rng = np.random.default_rng(random_state)
    return torchdata.Subset(dataset, rng.permutation(len(dataset)))

def split_dataset(dataset, class_imbalance, random_state):
    # class_imbalance p>=0.5 - Proportion of excess *even* labels on group 0
    rng = np.random.default_rng(random_state)
    generator = torch.manual_seed(rng.integers(10**6))
    ds_shuffle = shuffle_dataset(
        dataset=dataset,
        random_state=rng,
    )
    ds_y = np.array([y for x,y in ds_shuffle])
    ds_by_label = [
        torchdata.Subset(
            dataset=ds_shuffle,
            indices=(ds_y==y).nonzero()[0],
        )
        for y in range(10) # n_labels
    ]
    assert class_imbalance >= 0.5
    imbalance_p = [class_imbalance, 1-class_imbalance]
    ds_splits = [
        torchdata.random_split(
            dataset=ds,
            lengths=imbalance_p if y%2==0 else imbalance_p[::-1],
            generator=generator,
        )
        for y,ds in enumerate(ds_by_label)
    ]
    n_groups = len(ds_splits[0])
    groups = [
        torchdata.ConcatDataset(
            datasets=[ds_split[i] for ds_split in ds_splits],
        )
        for i in range(n_groups)
    ]
    groups_shuffle = [
        shuffle_dataset(
            dataset=ds,
            random_state=rng,
        )
        for ds in groups
    ]
    return groups_shuffle

def trim_dataset_size(dataset, maxsize, random_state):
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(dataset),size=maxsize,replace=False)
    assert len(set(indices))==maxsize
    return torchdata.Subset(
        dataset=dataset,
        indices=indices,
    )

def mix_datasets(datasets, p, random_state):
    rng = np.random.default_rng(random_state)
    total_size = min(len(ds) for ds in datasets)
    nvals = rng.multinomial(
        n=total_size,
        pvals=p,
    )
    return shuffle_dataset(
        dataset=torchdata.ConcatDataset(
            datasets=[
                trim_dataset_size(
                    dataset=ds,
                    maxsize=nval,
                    random_state=rng,
                )
                for ds,nval in zip(datasets,nvals)
            ]
        ),
        random_state=rng,
    )
                

def get_torchvision_dataset(dataset_name):
    return {
        dataset_type: getattr(torchvision.datasets,dataset_name)(
            root=torchvision_datasets_path,
            train=(dataset_type=='train'),
            download=True,
        )
        for dataset_type in ['train','test']
    }

