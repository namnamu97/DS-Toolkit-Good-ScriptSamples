from math_func import *
import pytest

@pytest.mark.parametrize(
    'arg1, arg2, result',
    [
        (7, 3, 10),
        ('Hello', ' World', 'Hello World'),
        (10.5, 25.5, 36)
    ]
)
def test_add(arg1, arg2, result):
    assert math_sum(arg1, arg2) == result