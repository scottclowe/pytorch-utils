'''
Tests for sampler.py.
'''

import torch

from .base_test import BaseTestCase, unittest
from .. import sampler


class TestLoopingRandomSampler(BaseTestCase):

    def test_length_actual(self):
        i = 0
        for repeat_state in (False, True):
            for n in (1, 2, 5, 7, 13, 100, 1000):
                for m in (1, 2, 5, 7, 13, 100, 1000):
                    i += 1
                    with self.subTest(i):
                        self.assertEqual(
                            m,
                            len(list(sampler.LoopingRandomSampler(torch.arange(n), m, repeat_state))),
                        )

    def test_length_reported(self):
        i = 0
        for repeat_state in (False, True):
            for n in (1, 2, 5, 7, 13, 100, 1000):
                for m in (1, 2, 5, 7, 13, 100, 1000):
                    i += 1
                    with self.subTest(i):
                        self.assertEqual(
                            m,
                            len(sampler.LoopingRandomSampler(torch.arange(n), m, repeat_state)),
                        )

    def test_repeated_cycle(self):
        out = list(sampler.LoopingRandomSampler(torch.arange(10), 100, True))
        for j in range(1, 10):
            self.assertEqual(out[:10], out[10 * j : 10 * (j + 1)])

    def test_not_repeated_cycle(self):
        torch.manual_seed(0)
        out = list(sampler.LoopingRandomSampler(torch.arange(10), 100, False))
        for j in range(1, 10):
            self.assertNotEqual(out[:10], out[10 * j : 10 * (j + 1)])

    def test_raises_float(self):
        with self.assert_raises(ValueError):
            sampler.LoopingRandomSampler(torch.arange(4), 3.5)

    def test_raises_nonpositive(self):
        with self.assert_raises(ValueError):
            sampler.LoopingRandomSampler(torch.arange(4), 0)

    def test_raises_nonbool(self):
        with self.assert_raises(ValueError):
            sampler.LoopingRandomSampler(torch.arange(4), 4, 7)
