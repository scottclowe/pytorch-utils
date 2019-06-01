import torch
from torch.utils.data.sampler import Sampler


class LoopingRandomSampler(Sampler):
    r"""Samples elements randomly, completing one full cycle of the dataset before starting
    to repeat itself.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw
        repeat_loop (bool): samples have the same order in each loop if ``True``, default=False
    """

    def __init__(self, data_source, num_samples, repeat_loop=False):
        self.data_source = data_source
        self.num_samples = num_samples
        self.repeat_loop = repeat_loop

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.repeat_loop, bool):
            raise ValueError("repeat_loop should be a boolean value, but got "
                             "repeat_loop={}".format(self.repeat_loop))

        if self.repeat_loop:
            self.order = torch.randperm(len(self.data_source)).tolist()

    def __iter__(self):
        n = len(self.data_source)
        i_sample = 0
        while i_sample < self.num_samples:
            for element in iter(self.order if self.repeat_loop else torch.randperm(n).tolist()):
                if i_sample >= self.num_samples:
                    break
                yield element
                i_sample += 1

    def __len__(self):
        return self.num_samples
