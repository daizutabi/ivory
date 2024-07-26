import torch.nn.functional as F

mse = F.mse_loss


def dense_cross_entropy(input, target, reduction="mean"):
    logprobs = F.log_softmax(input, 1)
    nll_loss = (-logprobs * target).sum(1)
    if reduction == "mean":
        return nll_loss.mean()
    return nll_loss
