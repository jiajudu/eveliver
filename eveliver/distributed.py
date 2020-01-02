import json

import torch


def distributed_broadcast(args, l):
    assert type(l) == list or type(l) == dict
    if args.local_rank < 0:
        return l
    else:
        torch.distributed.barrier()
        process_number = torch.distributed.get_world_size()
        json.dump(l, open(f'tmp/{args.local_rank}.json', 'w'))
        torch.distributed.barrier()
        objs = list()
        for i in range(process_number):
            objs.append(json.load(open(f'tmp/{i}.json')))
        if type(objs[0]) == list:
            ret = list()
            for i in range(process_number):
                ret.extend(objs[i])
        else:
            ret = dict()
            for i in range(process_number):
                for k, v in objs.items():
                    assert k not in ret
                    ret[k] = v
        torch.distributed.barrier()
        return ret
