from unittest import TestCase

import numpy as np

from spamneggs.numerics import optimum_from_step_sweep, valid_interval_from_step_sweep


class TestOptimumFromStepSweep(TestCase):
    def test_all_good(self):
        v = [0.9, 0.7, 0.5, 0.3, 0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]
        h = np.logspace(-8, -2, len(v))
        idx_opt, Δv, h_filtered, Δv_filtered = optimum_from_step_sweep(h, v)
        assert idx_opt == 4
        assert v[idx_opt] - v[idx_opt + 1] == 0.1
        assert np.all(h_filtered == h[:-1])

    def test_cancellation_first(self):
        v = [0.5, 0.5, 0.5, 0.2, 0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]
        h = np.logspace(-8, -2, len(v))
        idx_opt, Δv, h_filtered, Δv_filtered = optimum_from_step_sweep(h, v)
        assert idx_opt == 4
        assert v[idx_opt] - v[idx_opt + 1] == 0.1
        assert np.all(h_filtered == h[2:-1])

    def test_cancellation_middle(self):
        v = [0.9, 0.5, 0.5, 0.25, 0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]
        h = np.logspace(-8, -2, len(v))
        idx_opt, Δv, h_filtered, Δv_filtered = optimum_from_step_sweep(h, v)
        assert idx_opt == 4
        assert v[idx_opt] - v[idx_opt + 1] == 0.1

    def test_early_increase(self):
        v = [0.9, 0.7, 0.5, 0.3, 0.1, 0, 0.2, 0.4, 0.8, 1.5, 1]
        h = np.logspace(-8, -2, len(v))
        idx_opt, Δv, h_filtered, Δv_filtered = optimum_from_step_sweep(h, v)
        assert idx_opt == 4
        assert v[idx_opt] - v[idx_opt + 1] == 0.1


class TestValidIntervalFromStepSweep(TestCase):
    def test_all_good(self):
        e = [0, 0, 0, 0, 0, 0, 0]
        tol = 0.1
        b = valid_interval_from_step_sweep(e, tol)
        assert len(b) == 2
        assert b[0] == 0
        assert b[1] == len(e) - 1

    def test_two_bad_first(self):
        e = [1, 1, 0, 0, 0, 0, 0]
        tol = 0.1
        b = valid_interval_from_step_sweep(e, tol)
        assert len(b) == 2
        assert b[0] == 2
        assert b[1] == len(e) - 1

    def test_two_bad_middle(self):
        e = [0, 0, 1, 1, 0, 0, 0]
        tol = 0.1
        b = valid_interval_from_step_sweep(e, tol)
        assert len(b) == 2
        assert b[0] == 4
        assert b[1] == len(e) - 1

    def test_two_bad_last(self):
        e = [0, 0, 0, 0, 0, 1, 1]
        tol = 0.1
        b = valid_interval_from_step_sweep(e, tol)
        assert len(b) == 2
        assert b[0] == 0
        assert b[1] == 4
