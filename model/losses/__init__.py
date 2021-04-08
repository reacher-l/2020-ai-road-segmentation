import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_toolbelt import losses as L

from model.losses.pseudo_ce_loss import PseudoCrossEntropyLoss


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

        self.loss_func1 = nn.CrossEntropyLoss()
        self.loss_func2 = L.DiceLoss(mode='multiclass')

    def forward(self, logits, target):
        loss = self.loss_func1(logits[0], target) + 0.2 * self.loss_func2(logits[0], target)
        return loss


class SelfCorrectionLossFunction(nn.Module):
    def __init__(self, cycle=12):
        super(SelfCorrectionLossFunction, self).__init__()
        self.cycle = cycle

        self.sc_loss_func1 = PseudoCrossEntropyLoss()
        self.sc_loss_func2 = L.DiceLoss(mode='multiclass')

    def forward(self, predicts, target, soft_predict, cycle_n):
        with torch.no_grad:
            soft_predict = F.softmax(soft_predict, dim=1)
            soft_predict = self.weighted(self.to_one_hot(target, soft_predict.size(1)), soft_predict,
                                         alpha=1. / (cycle_n + 1))
        loss1 = self.sc_loss_func1(predicts[0], soft_predict)
        loss2 = self.sc_loss_func2(predicts, target)
        return loss1 + 0.2 * loss2

    @staticmethod
    def weighted(target_one_hot, soft_predict, alpha):
        soft_predict = alpha * target_one_hot + (1 - alpha) * soft_predict
        return soft_predict

    @staticmethod
    def to_one_hot(tensor, num_cls, dim=1, ignore_index=255):
        b, h, w = tensor.shape
        tensor[tensor == ignore_index] = 0
        onehot_tensor = torch.zeros(b, num_cls, h, w).cuda()
        onehot_tensor.scatter_(dim, tensor.unsqueeze(dim), 1)
        return onehot_tensor
