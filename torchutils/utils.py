import torch


def count_parameters(model, only_trainable=True):
    r"""
    Count the number of (trainable) parameters within a model and its children.

    Arguments:
        model (torch.nn.Model): the model.
        only_trainable (bool, optional): indicates whether the count should be restricted
            to only trainable parameters (ones which require grad), otherwise all
            parameters are included. Default is ``True``.

    Returns:
        int: total number of (trainable) parameters possessed by the model.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
