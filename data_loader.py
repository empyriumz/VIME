"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

data_loader.py
- Load and preprocess MNIST data (http://yann.lecun.com/exdb/mnist/)
"""

# Necessary packages
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

class UCIIncomeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_path, normalize=True):
        """
        Args:
            file_path (string): Path to load the csv file
            normalize: to normalize the dataset 
        """
        continous_col =  [
        "hours.per.week",
        "capital.loss",
        "fnlwgt",
        "age",
        "education.num",
        "capital.gain",
        ]
        self.data = pd.read_csv(file_path)
        for col in ['workclass', 'occupation', 'native.country']:
            self.data[col].replace('?', self.data[col].mode()[0], inplace=True)
        if normalize:
            max = pd.read_pickle("./data/max.pkl")
            min = pd.read_pickle("./data/min.pkl")
            self.data[continous_col] = (self.data[continous_col] - min) / (max - min + 1e-8)
        self.x = self.data.drop(['income'], axis=1)
        self.y = np.where(self.data['income'] == '<=50K', 0, 1)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'x': self.x.iloc[idx].values, 'y': self.y[idx]}
        #return list(self.x.iloc[idx]), self.y[idx]
        return sample


def uci_datasets(args, valid_size=0.2):
    """MNIST data loading.

    Args:
      - label_data_rate: ratio of labeled data

    Returns:
      - x_label, y_label: labeled dataset
      - x_unlab: unlabeled dataset
      - x_test, y_test: test dataset
    """
    # percentage of training set to use as validation
    valid_size = valid_size
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # choose the training and test datasets
    train_data = UCIIncomeDataset(file_path=args.train_file_path)
    
    test_data = UCIIncomeDataset(file_path=args.test_file_path)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    def collate_fn(data):
        x = [data[i]['x'] for i in range(len(data))]
        y = [data[i]['y'] for i in range(len(data))]
        return np.array(x), np.array(y)
    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                               collate_fn=collate_fn,
                                               sampler=train_sampler,
                                               **train_kwargs
                                               )
    
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               collate_fn=collate_fn,
                                               **train_kwargs,
                                               sampler=valid_sampler,
                                               )
    
    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_data,
                                              **test_kwargs
                                             )
    
    return train_loader, valid_loader, test_loader

def mnist_datasets(args, valid_size=0.2):
    """MNIST data loading.

    Args:
      - label_data_rate: ratio of labeled data

    Returns:
      - x_label, y_label: labeled dataset
      - x_unlab: unlabeled dataset
      - x_test, y_test: test dataset
    """
    # percentage of training set to use as validation
    valid_size = valid_size
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    # convert data to torch.FloatTensor and normalize
    transform = transforms.Compose(
        [
         transforms.ToTensor(), 
         transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    # choose the training and test datasets
    train_data = datasets.MNIST(root='data', 
                                train=True,
                                download=True, 
                                transform=transform)

    test_data = datasets.MNIST(root='data',
                               train=False,
                               download=True,
                               transform=transform)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                               sampler=train_sampler,
                                               **train_kwargs
                                               )
    
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               **train_kwargs,
                                               sampler=valid_sampler,
                                               )
    
    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_data,
                                              **test_kwargs
                                             )
    
    return train_loader, valid_loader, test_loader