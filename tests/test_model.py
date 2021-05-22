import pytest
import numpy as np
from src.model.model import Model

mdl = Model()

def test_function1():
    assert isinstance(mdl.function1(), int)
