import argparse
import importlib
from utils.utils import *
import yaml
import logging
PROJECT='pitsc'

def dict2namespace(dicts):
    for i in dicts:
        if isinstance(dicts[i], dict):
            dicts[i] = dict2namespace(dicts[i]) 
    ns = argparse.Namespace(**dicts)
    return ns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str,
                        choices=['FMC', 'nsynth-100', 'nsynth-200', 'nsynth-300', 'nsynth-400', 'librispeech',
                        'f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n', 'fsd'])
    parser.add_argument('-dataroot', type=str)
    parser.add_argument('-save_path', type=str)
    parser.add_argument('-config', type=str) 
    parser.add_argument('-debug', action='store_true')

    parser.add_argument('-lamda_proto', type=float)
    parser.add_argument('-way', type=int)
    parser.add_argument('-shot', type=int)
    parser.add_argument('-num_session', type=int)
    parser.add_argument('-pre_mixup_prob', type=float)
    parser.add_argument('-pre_mixup_alpha', type=float)
    parser.add_argument('-pre_cutmix_prob', type=float)
    parser.add_argument('-pre_cutmix_alpha', type=float)
    parser.add_argument('-pre_idty_prob', type=float)
    parser.add_argument('-pit_mixup_alpha', type=float)
    parser.add_argument('-seed', type=int)
    # about training
    parser.add_argument('-gpu', default='0')
    args = parser.parse_args()
    if args.pre_idty_prob is not None:
        args.pre_cutmix_prob = 1.0 - args.pre_idty_prob
    with open(args.config, 'r') as config:
        cfg = yaml.safe_load(config) 
    cfg = cfg['train']
    # cfg.update(vars(args))
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    args = argparse.Namespace(**cfg)
    args = dict2namespace(cfg)
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()