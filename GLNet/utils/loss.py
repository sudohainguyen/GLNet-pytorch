import torch.nn as nn
import torch.nn.functional as F
import torch

def one_hot(target, classes):
    # index is not flattened (pypass ignore) ############
    # size = index.size()[:1] + (classes,) + index.size()[1:]
    # view = index.size()[:1] + (1,) + index.size()[1:]
    # index is flatten (during ignore) ##################
    size = target.size()[:1] + (classes,)
    view = target.size()[:1] + (1,)

    # mask = torch.Tensor(size).fill_(0).to(device)
    mask = torch.zeros(size).cuda()
    index = target.view(view)
    ones = 1.0

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore

    def forward(self, input, target):
        """
        only support ignore at 0
        """
        classes = input.size(1)
        input = (
            input.permute(0, 2, 3, 1).contiguous().view(-1, classes)
        )  # B * H * W, C = P, C
        target = target.view(-1)
        if self.ignore is not None:
            valid = target != self.ignore
            input = input[valid]
            target = target[valid]

        if self.one_hot:
            target = one_hot(target, classes)
        probs = F.softmax(input, dim=1)
        probs = (probs * target).sum(1)
        probs = probs.clamp(self.eps, 1.0 - self.eps)

        log_p = probs.log()
        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            return batch_loss.mean()
        return batch_loss.sum()

class SoftCrossEntropyLoss2d(nn.Module):
    def forward(self, inputs, targets):
        loss = 0
        inputs = -F.log_softmax(inputs, dim=1)
        for index in range(inputs.size()[0]):
            loss += F.conv2d(
                inputs[range(index, index + 1)], targets[range(index, index + 1)]
            ) / (targets.size()[2] * targets.size()[3])
        return loss
