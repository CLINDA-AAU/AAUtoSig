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

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for p in group['params']:
            if p.gard is None:
                 continue
            grad = p.grad.data

            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            state['step'] += 1
            

            lr = group['lr']

            for layer in param_layers:
                                
                if not new_init:
                    for p_type, p in layer.items():
                        if clip_grad is not None:
                            p.grad.data.clamp_(-clip_grad, clip_grad)
                        self.update_param(p, lr, p_type, weight_reg)
                
                self.renormalize_layer(layer, u, norm_per)

        return loss