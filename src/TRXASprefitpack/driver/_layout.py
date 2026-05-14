"""
Helpers for mapping transient fitting parameters to array slices.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransientParamLayout:
    """Layout of a flat transient-fitting parameter vector.

    The layout describes how an optimization parameter vector is divided into
    IRF, time-zero, and lifetime-related parameter blocks.
    """

    num_irf: int
    num_t0: int
    num_tau: int

    def __post_init__(self) -> None:
        for name, value in (
            ("num_irf", self.num_irf),
            ("num_t0", self.num_t0),
            ("num_tau", self.num_tau),
        ):
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")

    @property
    def irf_slice(self) -> slice:
        return slice(0, self.num_irf)

    @property
    def t0_slice(self) -> slice:
        return slice(self.num_irf, self.num_irf + self.num_t0)

    @property
    def tau_slice(self) -> slice:
        start = self.num_irf + self.num_t0
        return slice(start, start + self.num_tau)

    @property
    def size(self) -> int:
        return self.num_irf + self.num_t0 + self.num_tau

    def unpack(self, x: np.ndarray):
        """Return IRF, t0, and tau blocks from a flat parameter vector."""
        x = np.asarray(x)

        if x.ndim != 1:
            raise ValueError("x must be a 1D parameter array.")
        if x.size != self.size:
            raise ValueError(
                f"x has size {x.size}, but this layout expects size {self.size}."
            )

        return x[self.irf_slice], x[self.t0_slice], x[self.tau_slice]