import random

import numpy as np
import torch
from tqdm import tqdm

from utils.argument import parse_argument
from utils.datasets import get_dataloader
from utils.model import get_model
from utils.loss import get_loss
from utils.optim import get_optimizer
from utils.saver import Saver
from utils.tools import get_gradient_norm, chunk_forward
from hessian_eigenthings import compute_hessian_eigenthings

def test_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


if __name__ == "__main__":
    args = parse_argument()

    # set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # handle saving results
    args.saver = Saver(args)

    # use gpu or not
    if args.cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # set learning rate
    args.lr = 2.0 / args.lr_denom

    # get dataset
    train_loader, test_loader, hessian_loader = get_dataloader(args)

    # create model
    model = get_model(args).to(args.device)

    criterion = get_loss(args)
    optimizer = get_optimizer(model, criterion, args)

    iteration = 0
    model.train()
    for epoch in range(args.epoch):
        for inputs, labels in train_loader:

            sharpness = None
            if args.eval_gap is None or iteration % args.eval_gap == 0:
                # calculate sharpness
                model.eval()
                power_iter_err = 1e-2
                sharpness, _ = compute_hessian_eigenthings(model, hessian_loader,
                        criterion, 1, use_gpu=args.cuda, full_dataset=True,
                        # max_samples=args.hessian_batch_size,
                        mode='power_iter',
                        power_iter_steps=20,
                        power_iter_err_threshold=power_iter_err)
                sharpness = sharpness[0]
                model.train()

            optimizer.zero_grad()
            outputs, sum_loss = chunk_forward(model, inputs, labels, criterion, args)
            if 'RK' in args.optim:
                optimizer.step(inputs, labels)
            else:
                optimizer.step()

            # calculate gradient norm
            norm = get_gradient_norm(model)

            # statistics
            loss = sum_loss.item()
            outputs = torch.cat(outputs, 0)
            _, prediction = torch.max(outputs, 1)
            # train acc
            train_acc = (prediction == labels).sum().item() / len(labels)
            # test acc
            test_acc = None
            if args.eval_gap is None or iteration % args.eval_gap == 0:
                model.eval()
                test_acc = test_model(model, test_loader)
                model.train()

            # write to file
            for k, v in zip(
                    ['loss', 'norm', 'train_acc'], [loss, norm, train_acc]):
                args.saver.write(k, iteration, v)

            # write to file
            if args.eval_gap is None or iteration % args.eval_gap == 0:
                for k, v in zip(
                        ['sharpness', 'test_acc'], [sharpness, test_acc]):
                    args.saver.write(k, iteration, v)

            iteration += 1
