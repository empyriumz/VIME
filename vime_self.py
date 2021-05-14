"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

vime_self.py
- Self-supervised learning parts of the VIME framework
- Using unlabeled data to train the encoder
"""

# Necessary packages

import torch.nn as nn
import torch.nn.functional as F
class VIME_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VIME_Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.fc_enc = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_mask = nn.Linear(self.hidden_dim, self.input_dim)
        self.fc_recon = nn.Linear(self.hidden_dim, self.input_dim)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, mask, x):
        mask = self.fc_enc(mask)
        mask = F.relu(mask)
        # apply sigmoid in the loss function later, for numerical stability
        mask = self.fc_mask(mask)
        x_enc = self.fc_enc(x)
        x_enc = F.relu(x_enc)
        x_recon = F.sigmoid(self.fc_recon(x_enc))
        return x_enc, mask, x_recon