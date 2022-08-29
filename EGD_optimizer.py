import torch

from tokenize import group
from torch.optim import Optimizer

class EGD_optim(Optimizer):
    '''
    Implements the Exponentiated gradient descent optimizer as introcuced
    by Kivinen & Warmuth 1997

    Parameters
    lr (float): learning rate. Default 1e-3
    '''
    def __init__(self, params, lr = 1e-3):
        if lr < 0.0:
            raise ValueError('Invalid learning rate: {} - should be >= 0.0'.format(lr))
        defaults = dict(lr = lr)
        super(EGD_optim, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(EGD_optim, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            #params_with_grad = []
            #d_p_list = []
            lr = group['lr']
            #has_sparse_grad = False

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                factor = torch.exp(d_p.mul_(-lr))

                p.data.mul_(factor)
                '''
                params_with_grad.append(p)
                d_p_list.append(p.grad) #gradient of the parameter
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]

            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                factor = torch.exp(d_p.mul_(-lr))

                param.mul_(factor)
            '''


        return loss

