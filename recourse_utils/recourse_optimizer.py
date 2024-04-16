import torch.optim as optim


def create_optimizer(params, cfg):
    params = filter(lambda p: p.requires_grad, params)
    if cfg.carma.params.optim_optimizer == 'adam':
        optimizer = optim.Adam(params, lr=float(cfg.carma.params.optim_lr),
                               weight_decay=float(cfg.carma.params.optim_weight_decay))
    elif cfg.carma.params.optim_optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=cfg.carma.params.optim_lr,
                              momentum=cfg.carma.params.optim_momentum,
                              weight_decay=cfg.carma.params.optim_weight_decay)
    else:
        raise ValueError('Optimizer {} not supported'.format(
            cfg.carma.params.optim_optimizer))

    return optimizer


def create_scheduler(optimizer, cfg):
    if cfg.carma.params.optim_scheduler == 'none':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=int(cfg.carma.params.optim_max_epochs) + 1)
    elif cfg.carma.params.optim_scheduler == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=cfg.carma.params.optim_gamma)
    elif cfg.carma.params.optim_scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=cfg.carma.params.optim_steps,
                                                   gamma=cfg.carma.params.optim_lr_decay)
    elif cfg.carma.params.optim_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=cfg.carma.params.optim_max_epoch)
    else:
        raise ValueError('Scheduler {} not supported'.format(
            cfg.carma.params.optim_scheduler))
    return scheduler
