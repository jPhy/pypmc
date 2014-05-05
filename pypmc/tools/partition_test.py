'Unit tests for integer partition'

import numpy as np
from ._partition import *

import unittest

class TestPartition(unittest.TestCase):
    def test_partition(self):
        N = 11
        k = 5
        target = [3, 2, 2, 2, 2]
        partition_result = partition(N, k)
        np.testing.assert_equal(partition_result, target)
