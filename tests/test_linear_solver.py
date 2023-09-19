import numpy as np
import pytest

from elexsolver.LinearSolver import LinearSolver


def test_fit():
    solver = LinearSolver()
    with pytest.raises(NotImplementedError):
        solver.fit(np.ndarray((5, 3)), np.ndarray((1, 3)))
