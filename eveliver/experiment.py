import json
import os
import random
import shutil
import sys

import IPython
import numpy as np
import torch


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def experiment_init(args):
    set_seed(args)
    if args.debug:
        sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)
    if args.local_rank == -1:
        args.device = torch.device('cuda')
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    if args.local_rank > 0:
        torch.distributed.barrier()
    if args.local_rank <= 0:
        if not os.path.exists('r'):
            os.mkdir('r')
        runs = os.listdir('r')
        i = max([int(c) for c in runs], default=-1) + 1
        os.mkdir(os.path.join('r', str(i)))
        src_names = [source for source in os.listdir() if source.endswith('.py')]
        for src_name in src_names:
            shutil.copy(src_name, os.path.join('r', str(i), src_name))
        os.chdir(os.path.join('r', str(i)))
        os.mkdir('output')
        os.mkdir('tmp')
    else:
        runs = os.listdir('r')
        i = max([int(c) for c in runs])
        os.chdir(os.path.join('r', str(i)))
    if args.local_rank == 0:
        torch.distributed.barrier()
    if not args.train:
        args.epoch = 1
    if args.local_rank <= 0:
        print(args)
    print(f'Process rank: {args.local_rank}, device: {args.device}, distributed training: {bool(args.local_rank != -1)}')
    json.dump(sys.argv, open('output/args.json', 'w'))


def experiment_finish():
    json.dump(True, open('output/f.json', 'w'))
