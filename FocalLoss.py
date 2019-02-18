import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0).cuda()
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1)).cuda()
        mask = Variable(mask, volatile=index.volatile).cuda()

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha

    def forward(self, input, target, reduction=None):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit)  # cross entropy
        loss = self.alpha * loss * (1 - logit) ** self.gamma  # focal loss
        if reduction:
            return loss.sum()
        else:
            return loss.mean()
