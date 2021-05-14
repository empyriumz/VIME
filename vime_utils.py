"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

vime_utils.py
- Various utility functions for VIME framework

(1) mask_generator: Generate mask vector for self and semi-supervised learning
(2) pretext_generator: Generate corrupted samples for self and semi-supervised learning
(3) perf_metric: prediction performances in terms of AUROC or accuracy
(4) convert_matrix_to_vector: Convert two dimensional matrix into one dimensional vector
(5) convert_vector_to_matrix: Convert one dimensional vector into one dimensional matrix
"""

# Necessary packages
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

def mask_generator(p_m, x):
    """Generate mask vector.

    Args:
      - p_m: corruption probability
      - x: feature matrix

    Returns:
      - mask: binary mask matrix
    """
    mask = np.random.binomial(1, p_m, x.shape)
    return mask


def pretext_generator(m, x):
    """Generate corrupted samples.

    Args:
      m: mask matrix
      x: feature matrix

    Returns:
      m_new: final mask matrix after corruption
      x_tilde: corrupted feature matrix
    """

    # Parameters
    no, dim = x.shape
    # Randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]

    # Corrupt samples
    x_tilde = x * (1 - m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde


def unsup_data_generator(p_m, x, device):
    """Generate masks and corrupted samples.

    Args:
      m: mask matrix
      x: feature matrix

    Returns:
      x_tilde: corrupted feature matrix
    """
    # Parameters
    no, dim = x.size()
    mask = p_m * torch.ones_like(x)
    mask = torch.bernoulli(mask)
    # Randomly (and column-wise) shuffle data
    x_shuffle = torch.zeros([no, dim])
    for i in range(dim):
        idx = torch.randperm(no)
        x_shuffle[:, i] = x[idx, i]
    # Corrupt samples
    x_tilde = x * (1 - mask) + x_shuffle * mask
    # re-calculate mask, in case the original value is sampled, i.e., no replacement
    mask = 1. * (x != x_tilde)
    return mask.to(device), x_tilde.to(device)


def perf_metric(metric, y_test, y_test_hat):
    """Evaluate performance.

    Args:
      - metric: acc or auc
      - y_test: ground truth label
      - y_test_hat: predicted values

    Returns:
      - performance: Accuracy or AUROC performance
    """
    # Accuracy metric
    if metric == "acc":
        result = accuracy_score(
            np.argmax(y_test, axis=1), np.argmax(y_test_hat, axis=1)
        )
    # AUROC metric
    elif metric == "auc":
        result = roc_auc_score(y_test[:, 1], y_test_hat[:, 1])

    return result


def convert_matrix_to_vector(matrix):
    """Convert two dimensional matrix into one dimensional vector

    Args:
      - matrix: two dimensional matrix

    Returns:
      - vector: one dimensional vector
    """
    # Parameters
    no, dim = matrix.shape
    # Define output
    vector = np.zeros(
        [
            no,
        ]
    )

    # Convert matrix to vector
    for i in range(dim):
        idx = np.where(matrix[:, i] == 1)
        vector[idx] = i

    return vector


def convert_vector_to_matrix(vector):
    """Convert one dimensional vector into two dimensional matrix

    Args:
      - vector: one dimensional vector

    Returns:
      - matrix: two dimensional matrix
    """
    # Parameters
    no = len(vector)
    dim = len(np.unique(vector))
    # Define output
    matrix = np.zeros([no, dim])

    # Convert vector to matrix
    for i in range(dim):
        idx = np.where(vector == i)
        matrix[idx, i] = 1

    return matrix

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Code source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(
        self, patience=4, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 4
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
