def test_arithmetic_is_broken():
    assert 1 + 1 == 3


def test_false_is_true():
    assert False


def test_pi_is_rational():
    import math

    assert math.pi == 22 / 7
