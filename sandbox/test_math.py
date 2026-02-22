import pytest
from math_utils import add, multiply, divide, fibonacci

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0

def test_divide():
    assert divide(10, 2) == 5.0
    assert divide(9, 3) == 3.0

def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(5, 0)

def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(2) == 1
    assert fibonacci(3) == 2
    assert fibonacci(4) == 3
    assert fibonacci(5) == 5
    assert fibonacci(6) == 8

def test_fibonacci_negative():
    with pytest.raises(ValueError, match="n must be non-negative"):
        fibonacci(-1)
