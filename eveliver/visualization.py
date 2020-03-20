import torch


def write_mean_std_max_min_absmean(writer, name, tensor, step):
    if tensor.shape[0] == 0:
        return
    name = 'zoo.' + name
    writer.add_scalar(name + 'mean', torch.mean(tensor).item(), global_step=step)
    writer.add_scalar(name + 'std', torch.std(tensor).item(), global_step=step)
    writer.add_scalar(name + 'max', torch.max(tensor).item(), global_step=step)
    writer.add_scalar(name + 'min', torch.min(tensor).item(), global_step=step)
    writer.add_scalar(name + 'absmean', torch.mean(torch.abs(tensor)).item(), global_step=step)
    writer.add_histogram(name + 'histogram', tensor.detach().cpu().data.numpy(), global_step=step)


def write_statistics(writer, model, step):
    for name, parameter in model.named_parameters():
        if parameter.dtype not in [torch.float, torch.double, torch.half]:
            continue
        if parameter.is_sparse:
            write_mean_std_max_min_absmean(writer, name + '/', parameter._values(), step)
        else:
            write_mean_std_max_min_absmean(writer, name + '/', parameter, step)
        if parameter.grad is not None:
            if parameter.grad.is_sparse:
                write_mean_std_max_min_absmean(writer, name + '/grad-', parameter.grad._values(), step)
            else:
                write_mean_std_max_min_absmean(writer, name + '/grad-', parameter.grad, step)
