import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class PseudoCrossEntropyLoss(nn.Module):
    def __init__(self, dim=1):
        super(PseudoCrossEntropyLoss, self).__init__()
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor):
        input_log_prob = F.log_softmax(input, dim=self.dim)
        loss = torch.sum(-input_log_prob * target, dim=self.dim)
        return loss.mean()
