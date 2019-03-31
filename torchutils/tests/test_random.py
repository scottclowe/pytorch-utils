'''
Tests for random.py.
'''

import random
import numpy as np
import torch

from .base_test import BaseTestCase, unittest
from ..random import seed_all


class TestSeedAll(BaseTestCase):

    def test_builtin(self):
        seed_all(0)
        x1 = random.randint(0, 1000000)
        x2 = random.randint(0, 1000000)
        self.assertTrue(x1 != x2)
        seed_all(1)
        x3 = random.randint(0, 1000000)
        self.assertTrue(x1 != x3)
        seed_all(0)
        x4 = random.randint(0, 1000000)
        self.assertTrue(x1 == x4)

    def test_numpy(self):
        seed_all(0)
        x1 = np.random.randint(0, 1000000)
        x2 = np.random.randint(0, 1000000)
        self.assertTrue(x1 != x2)
        seed_all(1)
        x3 = np.random.randint(0, 1000000)
        self.assertTrue(x1 != x3)
        seed_all(0)
        x4 = np.random.randint(0, 1000000)
        self.assertTrue(x1 == x4)

    def test_numpy2(self):
        seed_all(0)
        self.assert_equal(np.random.randint(1000), 684)
        seed_all(4294967295)
        self.assert_equal(np.random.randint(1000), 419)

    def test_torch_cpu(self):
        seed_all(0)
        x1 = torch.randint(0, 1000000, (1,))
        x2 = torch.randint(0, 1000000, (1,))
        self.assertTrue((x1 != x2).item())
        seed_all(1)
        x3 = torch.randint(0, 1000000, (1,))
        self.assertTrue((x1 != x3).item())
        seed_all(0)
        x4 = torch.randint(0, 1000000, (1,))
        self.assert_array_equal(x1, x4)

    def test_seed_none(self):
        seed_all(None)
        random.randint(0, 1000000)
        np.random.randint(0, 1000000)
        torch.randint(0, 1000000, (1,))
