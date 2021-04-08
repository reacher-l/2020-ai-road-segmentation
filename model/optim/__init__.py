import torch.optim as optim

from .radam import RAdam
from .lookahead import Lookahead
from .cyclicLR import CyclicCosAnnealingLR
from .warmup_scheduler import GradualWarmupScheduler


def get_optimizer(params, optimizer_cfg):
    if optimizer_cfg['mode'] == 'SGD':
        optimizer = optim.SGD(params, lr=optimizer_cfg['lr'], momentum=0.9,
                              weight_decay=optimizer_cfg['weight_decay'], nesterov=optimizer_cfg['nesterov'])
    elif optimizer_cfg['mode'] == 'RAdam':
        optimizer = RAdam(params, lr=optimizer_cfg['lr'], betas=(0.9, 0.999),
                          weight_decay=optimizer_cfg['weight_decay'])
    else:
        optimizer = optim.Adam(params, lr=optimizer_cfg['lr'], betas=(0.9, 0.999),
                               weight_decay=optimizer_cfg['weight_decay'])

    if optimizer_cfg['lookahead']:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)

    # todo: add split_weights.py

    return optimizer


def get_scheduler(optimizer, scheduler_cfg):
    MODE = scheduler_cfg['mode']

    if MODE == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                  max_lr=optimizer.param_groups[0]['lr'],
                                                  total_steps=scheduler_cfg['steps'],
                                                  pct_start=scheduler_cfg['pct_start'],
                                                  final_div_factor=scheduler_cfg['final_div_factor'],
                                                  cycle_momentum=scheduler_cfg['cycle_momentum'],
                                                  anneal_strategy=scheduler_cfg['anneal_strategy'])

    elif MODE == 'PolyLR':
        lr_lambda = lambda step: (1 - step / scheduler_cfg['steps']) ** scheduler_cfg['power']
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif MODE == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_cfg['steps'],
                                                         eta_min=scheduler_cfg['eta_min'])

    elif MODE == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   scheduler_cfg['milestones'],
                                                   gamma=scheduler_cfg['gamma'])

    elif MODE == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=scheduler_cfg['T_0'],
                                                                   T_mult=scheduler_cfg['T_multi'],
                                                                   eta_min=scheduler_cfg['eta_min'])

    elif MODE == 'CyclicCosAnnealingLR':
        scheduler = CyclicCosAnnealingLR(optimizer,
                                         milestones=scheduler_cfg['milestones'],
                                         decay_milestones=scheduler_cfg['decay_milestones'],
                                         eta_min=scheduler_cfg['eta_min'],
                                         gamma=scheduler_cfg['gamma'])

    elif scheduler_cfg.MODE == 'GradualWarmupScheduler':
        milestones = list(map(lambda x: x - scheduler_cfg['warmup_steps'], scheduler_cfg['milestones']))
        scheduler_steplr = optim.lr_scheduler.MultiStepLR(optimizer,
                                                          milestones=milestones,
                                                          gamma=scheduler_cfg['gamma'])
        scheduler = GradualWarmupScheduler(optimizer,
                                           multiplier=scheduler_cfg['milestones'],
                                           total_epoch=scheduler_cfg['warmup_steps'],
                                           after_scheduler=scheduler_steplr)
    else:
        raise ValueError

    return scheduler
