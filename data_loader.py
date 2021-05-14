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
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

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
        # cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
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