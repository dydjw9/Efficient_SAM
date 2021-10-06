import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_crossentropy(pred, gold, smoothing=0.0):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

def trades_loss(pred, gold,logits):
    loss1 = F.cross_entropy(pred,gold,reduction="mean")
    loss2 = F.kl_div(F.log_softmax(pred),F.softmax(logits),reduction="batchmean")
    loss = loss1 + 6 * loss2
    return loss

