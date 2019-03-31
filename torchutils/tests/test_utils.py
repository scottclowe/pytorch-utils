'''
Tests for utils.py.
'''

import torch.nn

from .base_test import BaseTestCase, unittest
from .. import utils


class TestCountParameters(BaseTestCase):

    def test_linear(self):
        m = torch.nn.Linear(4, 6, bias=False)
        expected = 24
        self.assertEqual(expected, utils.count_parameters(m))
        for include_flag in (True, False):
            actual = utils.count_parameters(m, only_trainable=include_flag)
            self.assertEqual(expected, actual)

        m = torch.nn.Linear(4, 6, bias=True)
        expected = 30
        self.assertEqual(expected, utils.count_parameters(m))
        for include_flag in (True, False):
            actual = utils.count_parameters(m, only_trainable=include_flag)
            self.assertEqual(expected, actual)

    def test_freeze(self):
        m = torch.nn.Linear(4, 6, bias=True)
        m.weight.requires_grad = False
        actual = utils.count_parameters(m, only_trainable=True)
        self.assertEqual(6, actual)
        actual = utils.count_parameters(m, only_trainable=False)
        self.assertEqual(30, actual)

    def test_dropout(self):
        m = torch.nn.Dropout()
        expected = 0
        self.assertEqual(expected, utils.count_parameters(m))
        for include_flag in (True, False):
            actual = utils.count_parameters(m, only_trainable=include_flag)
            self.assertEqual(expected, actual)

    def test_sequential(self):
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 6, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 6, bias=True),
        )
        expected = 54
        for include_flag in (True, False):
            actual = utils.count_parameters(m, only_trainable=include_flag)
            self.assertEqual(expected, actual)
