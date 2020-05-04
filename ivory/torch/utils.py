def cuda(x):
    if isinstance(x, list):
        return [xi.cuda() for xi in x]
    elif isinstance(x, dict):
        return {key: x[key].cuda() for key in x}
    else:
        return x.cuda()


def cpu(x):
    if isinstance(x, list):
        return [xi.cpu() for xi in x]
    elif isinstance(x, dict):
        return {key: x[key].cpu() for key in x}
    else:
        return x.cpu()
