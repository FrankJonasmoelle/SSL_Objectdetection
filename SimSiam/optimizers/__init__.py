# code from https://github.com/PatrickHua/SimSiam/blob/75a7c51362c30e8628ad83949055ef73829ce786/optimizers/lr_scheduler.py
import torch
from .lr_scheduler import LR_Scheduler


def get_optimizer(name, model, lr, momentum, weight_decay):

    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    },{
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]
   
    if name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
   
    return optimizer