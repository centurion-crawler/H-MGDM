import logging
import os
import random
import time
from glob import glob
import numpy as np
import torch
import torch.nn as nn
from model.GDM import GDM

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Use Xavier initialization for linear layers
        nn.init.zeros_(m.bias)  # Initialize biases with zeros

def load_partial_weights(model, state_dict):
    model_dict = model.state_dict()
    
    # 过滤掉不匹配的权重
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    
    # 更新模型的状态字典
    model_dict.update(filtered_dict)
    
    # 加载更新后的状态字典
    model.load_state_dict(model_dict)

    return model

def load_weights(w_path,Model_name="GDM",feature=False):
    ckpt_dict = torch.load(w_path,map_location="cpu")
    config = ckpt_dict['config']
    infer_model = GDM(config.model)
    if not feature:
        infer_model = load_partial_weights(infer_model, ckpt_dict['model'])
        # infer_model.load_state_dict(ckpt_dict['model'],strict=False)
    else:
        infer_model = load_partial_weights(infer_model, ckpt_dict['Diffmodel'])
        # infer_model.load_state_dict(ckpt_dict['Diffmodel'],strict=False)
    return infer_model, config


def get_logger(name, log_dir=None, log_fn='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_fn))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def get_optimizer(cfg, model_parameters_list):
    if cfg.type == 'adam':
        params_list = []
        for m_p in model_parameters_list:
            params_list.append({'params':m_p,'lr':cfg.lr,'weight_decay':cfg.weight_decay,'betas':(cfg.beta1, cfg.beta2,)})
        return torch.optim.Adam(
            params_list
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)



def get_scheduler(cfg, optimizer):
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
        )
    elif cfg.type == 'expmin':
        return ExponentialLRWithMinLr(
            optimizer,
            gamma=cfg.factor,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'expmin_milestone':
        gamma = np.exp(np.log(cfg.factor) / cfg.milestone)
        return ExponentialLRWithMinLr(
            optimizer,
            gamma=gamma,
            min_lr=cfg.min_lr,
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)