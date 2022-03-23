from math_func import *
import pytest

# @pytest.mark.number
def test_math_sum():
    assert math_sum(5,5) == 10

@pytest.mark.skip(reason = 'skip this one')
def test_math_product():
    assert math_product(5,5) == 25

# running this script normally: $ pytest test_math_func.py -v
# running all pytest scripts: $ py.test -v
# running a specific method: $ pytest test_math_func.py::test_math_sum -v 
# running assigned mark method: $ pytest test_math_func.py -v -m number