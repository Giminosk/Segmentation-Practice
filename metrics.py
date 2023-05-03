import torch
from torchmetrics import Dice, JaccardIndex, CosineSimilarity, Accuracy


def get_metrics(pred, target, device):
    B = target.shape[0]
    dice = Dice(average='macro', num_classes=B).to(device)
    iou = JaccardIndex(task='binary').to(device)
    cosine = CosineSimilarity(reduction = 'mean').to(device)
    acc = Accuracy(task='binary').to(device)

    dice_score = dice(pred, target)
    iou_score = iou(pred, target)
    cosine_score = cosine(pred, target)
    acc_score = acc(pred, target)

    return torch.Tensor([dice_score, iou_score, cosine_score, acc_score])


# def dice(pred, target, smooth=1e-8):
#     B = target.shape[0]
#     pred =  pred.view(B, -1)
#     target =  target.view(B, -1)

#     intersection = (pred * target).sum(1)
#     union = pred.sum(1) + target.sum(1)

#     dice = (2. * intersection) / (union + smooth)
#     return dice.sum() / B


# def accuracy(pred, target):
#     correct = torch.eq(pred, target).int()
#     return float(correct.sum()) / float(correct.numel())