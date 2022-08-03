from torch import optim
import torch
from torch.optim.optimizer import Optimizer
from utils.tools import chunk_forward
from copy import deepcopy

class RK(Optimizer):
    def __init__(self, model, criterion, args,
            betas=[1, 2, 2, 1]):
        self.params = list(model.parameters())
        self.model = model
        self.criterion = criterion
        self.args = args
        self.step_size = args.lr
        super(RK, self).__init__(self.params, {})

        self.betas = betas
        self.alpha = sum(betas)

    def step(self, inputs, labels):
        # x_{n+1} = x_n + (\eta / \alpha) * d
        # d = beta_1 * k_1 + beta_2 * k2 + ...
        # k_{i+1} = - \nabla f(x_n + (\eta / \beta_{i+1}) * k_i)
        k = []
        with torch.no_grad():
            for p in self.params:
                k.append(-1 * p.grad.data)
            d = deepcopy(k)
            for p in d:
                p.data = self.betas[0] * p

        for beta in self.betas[1: ]:
            temp_model = deepcopy(self.model)
            temp_model.zero_grad()
            with torch.no_grad():
                for p, p_k in zip(temp_model.parameters(), k):
                    p.add_(p_k, alpha=self.step_size / beta)
            # calculate gradient
            chunk_forward(temp_model, inputs, labels,
                    self.criterion, self.args)
            # update k and d
            with torch.no_grad():
                for p, p_k, p_d in zip(temp_model.parameters(), k, d):
                    p_k.data = -1 * p.grad.data
                    p_d.add_(p_k, alpha=beta)
        # update model param
        with torch.no_grad():
            for p, p_d in zip(self.params, d):
                p.add_(p_d, alpha=self.step_size / self.alpha)


def get_optimizer(model, criterion, args):
    if args.momentum == 0:
        args.nesterov = False
    if args.optim == 'GD' or args.optim == 'SGD':
        return optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                nesterov=args.nesterov)
    elif args.optim == 'Adam':
        return optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=args.adam_betas)
    elif args.optim == 'RMSprop':
        return optim.RMSprop(
                model.parameters(),
                lr=args.lr,
                alpha=args.rms_alpha,
                momentum=args.momentum)
    elif args.optim == 'RK4':
        return RK(model, criterion, args, betas=[1, 2, 2, 1])
    elif args.optim == 'RK2':
        return RK(model, criterion, args, betas=[1, 1])
    else:
        raise NotImplementedError
