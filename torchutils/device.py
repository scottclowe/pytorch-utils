import torch


def cuda_is_really_available():
    r"""
    Robustly checks to see if cuda is really available.

    We confirm the device is available by running dummy code on it,
    instead of trusting the output of `torch.cuda.is_available`.

    This protects against the situation where a CUDA GPU is present, but
    its CUDA capability is lower than your version of pytorch supports.
    In such a scenario, you will get a warning like:

        Found GPU0 GeForce XXX which is of cuda capability X.Y.
        PyTorch no longer supports this GPU because it is too old.

    and an error message

        RuntimeError: CUDA error: all CUDA-capable devices are busy or unavailable
    """
    if not torch.cuda.is_available():
        return False
    try:
        # Try to create a tensor on the GPU. To force it to be evaluated
        # immediately, we ask for the tensor to be converted into a string.
        # Otherwise nothing will happen since the computation is lazy.
        tmp = str(torch.zeros(2, device=torch.device('cuda')))
        return True
    except RuntimeError:
        return False
