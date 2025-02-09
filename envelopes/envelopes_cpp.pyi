from typing import Optional, List
import numpy as np


class VectorBuffer:
    ...


class WaveCache:
    def __init__(self, resolution: float) -> None:
        ...

    def resolution(self) -> float:
        ...


class AbstractEnvelope:
    ...


class Envelope:
    ...


class Gaussian(Envelope):
    def __init__(self, t0: float, w: float, amp: float) -> None:
        ...


class GaussianDRAG(Envelope):
    def __init__(self, t0: float, w: float, amp: float, coef: float, df: float,
                 phase: float) -> None:
        ...


class CosineDRAG(Envelope):
    def __init__(self, t0: float, w: float, amp: float, coef: float, df: float,
                 phase: float) -> None:
        ...


class Triangle(Envelope):
    def __init__(self, t0: float, tlen: float, amp: float, fall: bool) -> None:
        ...


class Rect(Envelope):
    def __init__(self, t0: float, tlen: float, amp: float) -> None:
        ...


class Flattop(Envelope):
    def __init__(self, t0: float, tlen: float, w_left: float, w_right: float,
                 amp: float) -> None:
        ...


class RippleRect(Envelope):
    def __init__(self, t0: float, tlen: float, amp: float, w: float,
                 ripple0: float, ripple1: float, ripple2: float,
                 ripple3: float) -> None:
        ...


class EnvSum(AbstractEnvelope):
    def start(self) -> float:
        ...

    def end(self) -> float():
        ...


def decode_envelope(env: AbstractEnvelope,
                    wc: WaveCache,
                    start: Optional[float] = None,
                    end: Optional[float] = None) -> tuple[float, VectorBuffer]:
    ...


def serialization(env: AbstractEnvelope) -> bytes:
    ...


def deserialization(data: bytes) -> AbstractEnvelope:
    ...


def align(env: AbstractEnvelope,
          dt: np.ndarray,
          amp: Optional[np.ndarray] = None) -> EnvSum:
    ...


def split(env: AbstractEnvelope, starts: np.ndarray,
          ends: np.ndarray) -> List[AbstractEnvelope]:
    ...


def mix(env1: AbstractEnvelope, df: float, phase: float,
        dynamical: bool) -> AbstractEnvelope:
    ...
