import torch


def tensor_to_obj(*tensors):
    assert len(tensors) > 0
    ret = list()
    for t in tensors:
        if len(t.shape) == 0:
            ret.append(t.detach().cpu().item())
        else:
            ret.append(t.detach().cpu().tolist())
    if len(ret) > 1:
        return tuple(ret)
    else:
        return ret[0]


def load_model(model, path):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    state_keys = list(state_dict.keys())
    for key in state_keys:
        if key.startswith('module.'):
            v = state_dict[key]
            del state_dict[key]
            state_dict[key[7:]] = v
    model.load_state_dict(state_dict)
