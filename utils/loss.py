import torch.nn as nn
import torch.nn.functional as F

def MSEcriterion(outputs, labels):
    labels = F.one_hot(labels, num_classes=10)
    loss = 0.5 * F.mse_loss(outputs, labels.float(), reduction='none')\
            .sum(dim=1).mean()
    return loss

def get_loss(args):
    if args.loss == 'CE':
        return nn.CrossEntropyLoss()
    elif args.loss == 'MSE':
        return MSEcriterion
