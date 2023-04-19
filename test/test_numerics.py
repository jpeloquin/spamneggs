from unittest import TestCase

from spamneggs.numerics import valid_interval_from_step_sweep


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
