import random

import numpy as np
import torch


def seed_all(seed=None, only_current_gpu=False, mirror_gpus=False):
    r"""
    Initialises the random number generators for random, numpy, and both CPU and GPU(s)
    for torch.

    Arguments:
        seed (int, optional): seed value to use for the random number generators.
            If :attr:`seed` is ``None`` (default), seeds are picked at random using
            the methods built in to each RNG.
        only_current_gpu (bool, optional): indicates whether to only re-seed the current
            cuda device, or to seed all of them. Default is ``False``.
        mirror_gpus (bool, optional): indicates whether all cuda devices should receive
            the same seed, or different seeds. If :attr:`mirror_gpus` is ``False`` and
            :attr:`seed` is not ``None``, each device receives a different but
            deterministically determined seed. Default is ``False``.

    Note that we override the settings for the cudnn backend whenever this function is
    called. If :attr:`seed` is not ``None``, we set::

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    in order to ensure experimental results behave deterministically and are repeatible.
    However, enabling deterministic mode may result in an impact on performance. See
    `link`_ for more details. If :attr:`seed` is ``None``, we return the cudnn backend
    to its performance-optimised default settings of::

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    .. _link:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    # Note that random, np.random and torch's RNG all have different
    # implementations so they will produce different numbers even with
    # when they are seeded the same.

    # Seed Python's built-in random number generator
    random.seed(seed)
    # Seed numpy's random number generator
    np.random.seed(seed)

    def get_seed():
        '''
        On Python 3.2 and above, and when system sources of randomness are
        available, use `os.urandom` to make a new seed. Otherwise, use the
        current time.
        '''
        try:
            import os
            # Use system's source of entropy (on Linux, syscall `getrandom()`)
            s = int.from_bytes(os.urandom(4), byteorder="little")
        except AttributeError:
            from datetime import datetime
            # Get the current time in mircoseconds, and map to an integer
            # in the range [0, 2**32)
            s = int((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds()
                    * 1000000) % 4294967296
        return s

    # Seed pytorch's random number generator on the CPU
    # torch doesn't support a None argument, so we have to source our own seed
    # with high entropy if none is given.
    s = seed if seed is not None else get_seed()
    torch.manual_seed(s)

    if seed is None:
        # Since seeds are random, we don't care about determinism and
        # will set the backend up for optimal performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        # Ensure cudNN is deterministic, so the results are consistent
        # for this seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Seed pytorch's random number generator on the GPU(s)
    if only_current_gpu:
        # Only re-seed the current GPU
        if mirror_gpus:
            # ... re-seed with the same as the CPU seed
            torch.cuda.manual_seed(s)
        elif seed is None:
            # ... re-seed at random, however pytorch deems fit
            torch.cuda.seed()
        else:
            # ... re-seed with a deterministic seed based on, but
            # not equal to, the CPU seed
            torch.cuda.manual_seed((seed + 1) % 4294967296)
    elif mirror_gpus:
        # Seed multiple GPUs, each with the same seed
        torch.cuda.manual_seed_all(s)
    elif seed is None:
        # Seed multiple GPUs, all with unique seeds
        # ... a random seed for each GPU, however pytorch deems fit
        torch.cuda.seed_all()
    else:
        # Seed multiple GPUs, all with unique seeds
        # ... different deterministic seeds for each GPU
        # We assign the seeds in ascending order, and can't exceed the
        # random state's maximum value of 2**32 == 4294967296
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                torch.cuda.manual_seed((seed + 1 + device) % 4294967296)
