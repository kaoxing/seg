import torch
from torch import nn

def diceCoeffv2(pred, gt, eps=1, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N


class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, activation='sigmoid'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeffv2(y_pr, y_gt, activation=self.activation)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.bce = nn.BCELoss(reduce=False)

    def forward(self, inputs, targets):
        BCELoss = self.bce(inputs, targets)
        pt = torch.exp(-BCELoss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCELoss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss