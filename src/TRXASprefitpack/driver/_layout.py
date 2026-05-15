"""
Helpers for mapping transient fitting parameters to array slices.
"""

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

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

@dataclass(frozen=True)
class DampedOscillationParamLayout:
    """
    Parameter layout for damped oscillation transient fitting.

    Parameter vector structure:
        [IRF params | t0 params | damping lifetimes | oscillation periods]
    """

    num_irf: int
    num_t0: int
    num_osc: int

    @property
    def irf_slice(self) -> slice:
        return slice(0, self.num_irf)

    @property
    def t0_slice(self) -> slice:
        return slice(self.num_irf, self.num_irf + self.num_t0)

    @property
    def tau_osc_slice(self) -> slice:
        start = self.num_irf + self.num_t0
        return slice(start, start + self.num_osc)

    @property
    def period_osc_slice(self) -> slice:
        start = self.num_irf + self.num_t0 + self.num_osc
        return slice(start, start + self.num_osc)

    @property
    def size(self) -> int:
        return self.num_irf + self.num_t0 + 2 * self.num_osc

    def unpack(
        self,
        x: Sequence[Any],
    ) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]]:
        """Unpack flat parameters into (irf, t0, tau_osc, period_osc)."""
        if len(x) < self.size:
            raise ValueError(f"Expected array of size at least {self.size}, got {len(x)}")

        return (
            x[self.irf_slice],
            x[self.t0_slice],
            x[self.tau_osc_slice],
            x[self.period_osc_slice],
        )

@dataclass(frozen=True)
class BothTransientParamLayout:
    """
    Parameter layout for combined decay + damped oscillation transient fitting.

    Parameter vector structure:
        [IRF params | t0 params | decay lifetimes | damping lifetimes | oscillation periods]
    """

    num_irf: int
    num_t0: int
    num_decay: int
    num_osc: int

    @property
    def irf_slice(self) -> slice:
        return slice(0, self.num_irf)

    @property
    def t0_slice(self) -> slice:
        return slice(self.num_irf, self.num_irf + self.num_t0)

    @property
    def tau_decay_slice(self) -> slice:
        start = self.num_irf + self.num_t0
        return slice(start, start + self.num_decay)

    @property
    def tau_osc_slice(self) -> slice:
        start = self.num_irf + self.num_t0 + self.num_decay
        return slice(start, start + self.num_osc)

    @property
    def period_osc_slice(self) -> slice:
        start = self.num_irf + self.num_t0 + self.num_decay + self.num_osc
        return slice(start, start + self.num_osc)

    @property
    def size(self) -> int:
        return self.num_irf + self.num_t0 + self.num_decay + 2 * self.num_osc

    def unpack(
        self,
        x: Sequence[Any],
    ) -> Tuple[
        Sequence[Any],
        Sequence[Any],
        Sequence[Any],
        Sequence[Any],
        Sequence[Any],
    ]:
        """Unpack flat parameters into (irf, t0, tau_decay, tau_osc, period_osc)."""
        if len(x) < self.size:
            raise ValueError(f"Expected array of size at least {self.size}, got {len(x)}")

        return (
            x[self.irf_slice],
            x[self.t0_slice],
            x[self.tau_decay_slice],
            x[self.tau_osc_slice],
            x[self.period_osc_slice],
        )