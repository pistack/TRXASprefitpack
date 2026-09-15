"""Shared pytest configuration."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def deterministic_numpy_random_state():
    """Make numerical tests reproducible across runs and CI shards."""
    np.random.seed(0)
