# Control envelopes in time domain

from typing import Iterable, Optional, List, Tuple, Union
from abc import ABC, abstractmethod

import math
import numpy as np
from scipy.special import erf


class LazyArray:
    '''
    To avoid repeated multiplication on array.
    NOTE this implementation differs from that in "envelopes_cpp" which is more efficient.
    '''
    __slots__ = ('c', 'array')

    def __init__(self, c: Union[float, complex], array: np.ndarray) -> None:
        self.c = c
        self.array = array

    def __imul__(
            self, other: Union[int, float, complex,
                               'LazyArray']) -> 'LazyArray':
        if isinstance(other, (int, float, complex)):
            self.c *= other
        elif isinstance(other, np.ndarray):
            self.array *= other
        elif isinstance(other, LazyArray):
            self.c *= other.c
            self.array *= other.array
        return self

    def __mul__(
        self, other: Union[int, float, complex, np.ndarray,
                           'LazyArray']) -> 'LazyArray':
        if isinstance(other, (int, float, complex)):
            return LazyArray(self.c * other, self.array)
        elif isinstance(other, np.ndarray):
            return LazyArray(self.c, self.array * other)
        elif isinstance(other, LazyArray):
            return LazyArray(self.c * other.c, self.array * other.array)
        else:
            raise TypeError(type(other))

    __rmul__ = __mul__

    def eval(self) -> np.ndarray:
        return self.c * self.array


TIME_ATOL = 1e-5
FREQ_ATOL = 1e-5
FLOAT_ATOL = 1e-10

END_PADDING = 2  # END_PADDING * WaveCache.resolution padding at end time


class WaveCache:
    '''
    Time domain cache for high-performance envelope calculation.
    cache_characteristic, [Tuple]) is used to identify if the cached data can be reused.
    cache_characteristic (class_name, base_start, args)
    base_start = start%DAC_resolution
    '''
    def __init__(self, resolution: float = 0.5, precision='single'):
        self.resolution = resolution
        self.resolution_over_atol = round(self.resolution / TIME_ATOL)
        self._cache = {}
        if precision == 'single':
            self.float_dtype = 'float32'
            self.complex_dtype = 'complex64'
        elif precision == 'double':
            self.float_dtype = 'float64'
            self.complex_dtype = 'complex128'
        else:
            raise NotImplementedError(precision)

    def get_cache(self, class_name: str,
                  start: float) -> Tuple[float, float, dict]:
        '''
        start = shift + base_start*TIME_ATOL
        '''
        start_over_atol = round(start / TIME_ATOL)
        base_start = start_over_atol % self.resolution_over_atol
        shift_t = start_over_atol // self.resolution_over_atol * self.resolution
        return base_start * TIME_ATOL, shift_t, self._cache.setdefault(
            class_name, {}).setdefault(base_start, {})


class AbstractEnvelope(ABC):
    '''
    Represents a control envelope as a function of time.
    Start and end time will be used to calculate effective wave data(nonzero).
    '''
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end
        self.is_complex = False

    @abstractmethod
    def time_func(self,
                  wc: WaveCache,
                  dt=0.0) -> Tuple[float, Union[np.ndarray, LazyArray]]:
        '''
        Returns the start time and waveform data of the envelope.
        '''
        pass

    def duration(self):
        '''
        Returns the duration of the envelope.
        '''
        return self.end - self.start

    def __add__(self, other):
        if isinstance(other, AbstractEnvelope):
            return EnvSum(self, other)
        elif other == 0:
            return self
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, AbstractEnvelope):
            return EnvSum(self, -other)
        elif other == 0:
            return self
        else:
            raise NotImplementedError

    def __rsub__(self, other):
        if isinstance(other, AbstractEnvelope):
            return EnvSum(-self, other)
        elif other == 0:
            return -self
        else:
            raise NotImplementedError

    def __mul__(self, other):
        return EnvProd(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, AbstractEnvelope):
            raise NotImplementedError
        else:
            return 1.0 / other * self

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

    def __rshift__(self, other):
        return EnvShift(self, other)


class EnvSum(AbstractEnvelope):
    '''
    Helper class to support __add__ and __sub__
    '''
    def __init__(self, *envelopes: Iterable[AbstractEnvelope]):
        self.items: List[AbstractEnvelope] = []
        for env in envelopes:
            if isinstance(env, EnvSum):
                self.items += env.items
            elif isinstance(env, AbstractEnvelope):
                self.items.append(env)
            else:
                raise TypeError(type(env))

    @property
    def start(self):
        starts = [env.start for env in self.items if env.start is not None]
        return min(starts) if len(starts) else None

    @property
    def end(self):
        ends = [env.end for env in self.items if env.end is not None]
        return max(ends) if len(ends) else None

    @property
    def is_complex(self):
        return any([_env.is_complex for _env in self.items])

    def __iadd__(self, other):
        if isinstance(other, EnvSum):
            self.items += other.items
        elif isinstance(other, AbstractEnvelope):
            self.items.append(other)
        else:
            raise NotImplementedError
        return self

    def __mul__(self, other):
        _env = EnvSum()
        _env.items = [item * other for item in self.items]
        return _env

    def __rshift__(self, other):
        _env = EnvSum()
        _env.items = [item >> other for item in self.items]
        return _env

    def time_func(self, wc: WaveCache, dt=0.0) -> Tuple[float, np.ndarray]:
        waves = []
        waves_start = []
        waves_end = []
        for item in self.items:
            _t, _wave = item.time_func(wc, dt=dt)
            if isinstance(_wave, LazyArray):
                _wave = _wave.eval()
            waves.append(_wave)
            waves_start.append(_t)
            waves_end.append(_t + len(_wave) * wc.resolution)
        waves_start = np.array(waves_start)
        waves_end = np.array(waves_end)
        start, end = np.min(waves_start), np.max(waves_end)
        sample_num = round((end - start) / wc.resolution)
        if self.is_complex:
            wave = np.zeros(sample_num, dtype=wc.complex_dtype)
        else:
            wave = np.zeros(sample_num, dtype=wc.float_dtype)
        waves_start_idx = np.round(
            (np.array(waves_start) - start) / wc.resolution).astype(int)
        for _start_idx, _wave in zip(waves_start_idx, waves):
            wave[_start_idx:_start_idx + len(_wave)] += _wave
        return start, wave


class EnvProd(AbstractEnvelope):
    '''
    Helper class to support __mul__, __div__, and __neg__.  Represents multiplication by a scalar
    '''
    def __init__(self, env: AbstractEnvelope, const: Union[float, complex]):
        self.env = env
        self.const = const

    @property
    def start(self):
        return self.env.start

    @property
    def end(self):
        return self.env.end

    @property
    def is_complex(self):
        return self.env.is_complex or isinstance(self.const, complex)

    def __imul__(self, other):
        self.const *= other
        return self

    def __mul__(self, other):
        return EnvProd(self.env, self.const * other)

    def __rshift__(self, other):
        return EnvProd(self.env >> other, self.const)

    def time_func(self,
                  wc: WaveCache,
                  dt=0.0) -> Tuple[float, Union[np.ndarray, LazyArray]]:
        t, wave = self.env.time_func(wc, dt=dt)
        return t, wave * self.const


class EnvShift(AbstractEnvelope):
    """Shift the envelope by dt."""
    def __init__(self, env: AbstractEnvelope, dt: float):
        self.dt = dt
        self.env = env
        self.start, self.end = self.env.start + dt, self.env.end + dt
        self.is_complex = self.env.is_complex

    def __rshift__(self, other):
        return EnvShift(self.env, self.dt + other)

    def time_func(self,
                  wc: WaveCache,
                  dt=0.0) -> Tuple[float, Union[np.ndarray, LazyArray]]:
        return self.env.time_func(wc, dt=dt + self.dt)


# ------- mixing envelopes --------


class MixExp(AbstractEnvelope):
    '''
    Design for mixing envelope, with frequency df and phase shift.
    DO NOT MODIFY THIS CLASS OR USE IT DIRECTLY.
    '''
    def __init__(self, start, end, df, phase=0.0, dynamical=True):
        """complex exp func for mix."""
        self.df = df
        self.phase = phase
        self.start, self.end = start, end
        self.is_complex = True
        self.dynamical = dynamical
        self.cache_characteristic = (round(self.df / FREQ_ATOL),
                                     round(self.duration() / TIME_ATOL))

    def time_func(self, wc: WaveCache, dt=0.0) -> Tuple[float, LazyArray]:
        _cache = wc._cache.setdefault(self.__class__.__name__,
                                      {}).setdefault('None', {})
        if self.cache_characteristic not in _cache:
            base_t = np.arange(0,
                               self.duration() + END_PADDING * wc.resolution -
                               wc.resolution / 2,
                               wc.resolution,
                               dtype=wc.float_dtype)
            _cache[self.cache_characteristic] = np.exp(-2j * np.pi * self.df *
                                                       base_t)
        base_wave = _cache[self.cache_characteristic]
        if self.dynamical:  # keep phase = 0 at t=0
            shift_t = round(
                (self.start + dt) /
                TIME_ATOL) // wc.resolution_over_atol * wc.resolution
            phase = -2 * np.pi * self.df * shift_t + self.phase
            return shift_t, LazyArray(
                math.cos(phase) + 1j * math.sin(phase), base_wave)
        else:  # keep phase = 0 at t=dt
            shift_t = round(
                (self.start + dt) /
                TIME_ATOL) // wc.resolution_over_atol * wc.resolution
            phase = -2 * np.pi * self.df * (shift_t - dt) + self.phase
            return shift_t, LazyArray(
                math.cos(phase) + 1j * math.sin(phase), base_wave)


class EnvMix(AbstractEnvelope):
    """Apply sideband mixing at difference frequency df."""
    def __init__(self, env: AbstractEnvelope, df, phase=0.0, dynamical=True):
        self.env = env
        self._exp = MixExp(self.env.start, self.env.end, df, phase, dynamical)
        self.start, self.end = self.env.start, self.env.end
        self.is_complex = True

    def time_func(self, wc: WaveCache, dt=0.0) -> Tuple[float, np.ndarray]:
        t, wave = self.env.time_func(wc, dt=dt)
        _, wave_exp = self._exp.time_func(wc, dt=dt)
        return t, wave_exp * wave


def mix(env, df, phase=0.0, dynamical=True):
    if isinstance(env, EnvSum):
        return EnvSum(*[
            mix(item, df=df, phase=phase, dynamical=dynamical)
            for item in env.items
        ])
    elif isinstance(env, EnvProd):
        return EnvProd(mix(env.env, df=df, phase=phase, dynamical=dynamical),
                       env.const)
    else:
        return EnvMix(env, df=df, phase=phase, dynamical=dynamical)


# ------- instance of specific Envelopes -----
class Envelope(AbstractEnvelope):
    def __init__(self,
                 start: float,
                 end: float,
                 cache_characteristic,
                 amp: Optional[Union[None, float]] = None):
        super().__init__(start, end)
        self._amp = amp
        self.cache_characteristic = cache_characteristic

    @abstractmethod
    def model(self, base_start: float, base_t: np.ndarray):
        pass

    def time_func(self,
                  wc: WaveCache,
                  dt=0.0) -> Tuple[float, Union[np.ndarray, LazyArray]]:
        start = self.start + dt
        base_start, shift_t, _cache = wc.get_cache(self.__class__.__name__,
                                                   start)
        if self.cache_characteristic not in _cache:
            base_t = np.arange(0,
                               self.duration() + END_PADDING * wc.resolution -
                               wc.resolution / 2,
                               wc.resolution,
                               dtype=wc.float_dtype)
            _cache[self.cache_characteristic] = self.model(base_start, base_t)
        base_wave = _cache[self.cache_characteristic]
        if self._amp is not None:
            base_wave = LazyArray(self._amp, base_wave)
        return shift_t, base_wave


class Gaussian(Envelope):
    def __init__(self, t0, w, amp=1.0):
        """
        A gaussian pulse with specified center and full-width at half max.
        """
        self.w = w
        # convert fwhm to std. deviation
        self.sigma = w / math.sqrt(8 * math.log(2))
        super().__init__(start=t0 - 3 * w,
                         end=t0 + 3 * w,
                         cache_characteristic=(round(self.w / TIME_ATOL), ),
                         amp=amp)

    def model(self, base_start: float, base_t: np.ndarray):
        base_t0 = base_start + 3 * self.w  # center of gaussian pulse is at base_t0
        wave = np.exp(-(base_t - base_t0)**2 / (2 * self.sigma**2))
        return wave


class GaussianDRAG(Envelope):
    def __init__(self,
                 t0: float,
                 w: float,
                 coef: float,
                 df: float,
                 amp: float = 1.0,
                 phase: float = 0.0):
        '''
        DRAG: PRL 103, 110501 (2009)
        A gaussian-DRAG pulse with specified center and full-width at half max.
        The DRAG coefficient is given by coef.
        phase is the initial phase of the pulse at t=0.
        '''
        self.w = w
        # convert fwhm to std. deviation
        self.sigma = w / math.sqrt(8 * math.log(2))
        self.coef = coef  # DRAG Q coef
        self.df = df
        self.phase = phase
        super().__init__(start=t0 - 3 * w,
                         end=t0 + 3 * w,
                         cache_characteristic=(round(self.w / TIME_ATOL),
                                               round(self.coef / FLOAT_ATOL),
                                               round(self.df / FREQ_ATOL)))
        self.amp = amp
        self.is_complex = True

    def model(self, base_start: float, base_t: np.ndarray):
        base_t0 = base_start + 3 * self.w  # center of gaussian pulse is at base_t0
        return np.exp(-(base_t - base_t0)**2 / (2 * self.sigma**2)) * (
            1 - 1j * self.coef * (base_t - base_t0) /
            (self.sigma**2)) * np.exp(-2j * np.pi * self.df * base_t)

    def time_func(self, wc: WaveCache, dt=0) -> Tuple[float, LazyArray]:
        t, wave = super().time_func(wc, dt)
        phase = -2 * np.pi * self.df * t + self.phase
        amp = self.amp * (math.cos(phase) + 1j * math.sin(phase))
        return t, LazyArray(amp, wave)


class CosineDRAG(Envelope):
    def __init__(self,
                 t0: float,
                 w: float,
                 coef: float,
                 df: float,
                 amp=1.0,
                 phase: float = 0.0):
        '''
        DRAG: PRL 103, 110501 (2009)
        A cosine-DRAG derivative function centered at t0 with FULL WIDTH w.
        The DRAG coefficient is given by coef.
        phase is the initial phase of the pulse at t=0.
        '''
        self.w = w
        self.coef = coef  # DRAG Q coef
        self.df = df
        self.phase = phase
        super().__init__(start=t0 - w / 2.0,
                         end=t0 + w / 2.0,
                         cache_characteristic=(round(self.w / TIME_ATOL),
                                               round(self.coef / FLOAT_ATOL),
                                               round(self.df / FREQ_ATOL)))
        self.amp = amp
        self.is_complex = True

    def model(self, base_start: float, base_t: np.ndarray):
        base_t0 = base_start + self.w / 2.0  # center of cosine pulse is at base_t0
        return (0.5 * (1 + np.cos(2 * np.pi * (base_t - base_t0) / self.w)) -
                1j * self.coef * np.pi / self.w *
                np.sin(2 * np.pi * (base_t - base_t0) / self.w)) * ((
                    (base_t - base_t0) + self.w / 2.) > 0) * (
                        (-(base_t - base_t0) + self.w / 2.) > 0) * np.exp(
                            -2j * np.pi * self.df * base_t)

    def time_func(self, wc: WaveCache, dt=0) -> Tuple[float, LazyArray]:
        t, wave = super().time_func(wc, dt)
        phase = -2 * np.pi * self.df * t + self.phase
        amp = self.amp * (math.cos(phase) + 1j * math.sin(phase))
        return t, LazyArray(amp, wave)


class Triangle(Envelope):
    def __init__(self, t0, tlen, amp=1.0, fall=True):
        """A triangular pulse, either rising or falling."""
        self.tlen = tlen
        self.fall = fall
        super().__init__(start=t0,
                         end=t0 + tlen,
                         amp=amp,
                         cache_characteristic=(round(self.tlen / TIME_ATOL),
                                               self.fall))

    def model(self, base_start: float, base_t: np.ndarray):
        if self.fall:
            base_wave = (base_t >=
                         base_start) * (base_t < self.tlen + base_start) * (
                             1 - (base_t - base_start) / self.tlen)
        else:
            base_wave = (base_t >= base_start) * (
                base_t < self.tlen + base_start) * (base_t -
                                                    base_start) / self.tlen
        return base_wave


class Rect(Envelope):
    def __init__(self, t0, tlen, amp=1.0):
        """A rectangular pulse with sharp turn on and turn off.
        """
        self.amp = amp
        self.tlen = tlen
        super().__init__(start=t0,
                         end=t0 + tlen,
                         amp=amp,
                         cache_characteristic=(round(self.tlen / TIME_ATOL), ))

    def model(self, base_start: float, base_t: np.ndarray):
        return (base_t >= base_start) * (base_t < self.tlen + base_start)


class Flattop(Envelope):
    def __init__(self, t0, tlen, w_left, w_right, amp=1.0):
        self.tlen = tlen
        self.w_left = w_left
        self.a_left = 2 * math.sqrt(math.log(2)) / self.w_left
        self.w_right = w_right
        self.a_right = 2 * math.sqrt(math.log(2)) / self.w_right
        super().__init__(start=t0 - 3 * self.w_left,
                         end=t0 + tlen + 3 * self.w_right,
                         amp=amp,
                         cache_characteristic=(round(self.w_left / TIME_ATOL),
                                               round(self.w_right / TIME_ATOL),
                                               round(self.tlen / TIME_ATOL)))

    def model(self, base_start: float, base_t: np.ndarray):
        base_t0 = base_start + 3 * self.w_left
        return (erf(self.a_right * (base_t0 + self.tlen - base_t)) -
                erf(self.a_left * (base_t0 - base_t))) / 2.0


class RippleRect(Envelope):
    def __init__(self, t0, tlen, amp, w, ripple0, ripple1, ripple2, ripple3):
        self.tlen = tlen
        self.ripples = [ripple0, ripple1, ripple2, ripple3]
        self.w = w
        super().__init__(
            start=t0 - 3 * self.w,
            end=t0 + tlen + 3 * self.w,
            amp=amp,
            cache_characteristic=(round(self.ripples[0] / FLOAT_ATOL),
                                  round(self.ripples[1] / FLOAT_ATOL),
                                  round(self.ripples[2] / FLOAT_ATOL),
                                  round(self.ripples[3] / FLOAT_ATOL),
                                  round(self.w / TIME_ATOL),
                                  round(self.tlen / TIME_ATOL)))

    def model(self, base_start: float, base_t: np.ndarray):
        base_t0 = base_start + 3 * self.w
        base_tmin = base_t0
        base_tmax = base_t0 + self.tlen
        base_tmid = (base_tmin + base_tmax) / 2
        base_amp = 1 - self.ripples[1] - self.ripples[3]
        sigma = (self.w + 1e-10
                 ) / 2 / math.log(2)**0.5  # NOT the sigma of Gaussian func
        base_wave = base_amp / 2 * (erf((base_t - base_tmin) / sigma) - erf(
            (base_t - base_tmax) / sigma))
        for idx, r in enumerate(self.ripples):
            idx_r = 2**(idx // 2)
            if np.mod(idx, 2) == 0:
                base_wave += r / 2 * np.exp(
                    -(idx_r * np.pi * sigma / 2 / self.tlen)**2) * np.imag(
                        np.exp(1j * idx_r * np.pi *
                               (base_t - base_tmid) / self.tlen) *
                        (erf((base_t - base_tmin) / sigma +
                             1j * idx_r * np.pi * sigma / 2 / self.tlen) - erf(
                                 (base_t - base_tmax) / sigma +
                                 1j * idx_r * np.pi * sigma / 2 / self.tlen)))
            if np.mod(idx, 2) == 1:
                base_wave += r / 2 * np.exp(
                    -(idx_r * np.pi * sigma / 2 / self.tlen)**2) * np.real(
                        np.exp(1j * idx_r * np.pi *
                               (base_t - base_tmid) / self.tlen) *
                        (erf((base_t - base_tmin) / sigma +
                             1j * idx_r * np.pi * sigma / 2 / self.tlen) - erf(
                                 (base_t - base_tmax) / sigma +
                                 1j * idx_r * np.pi * sigma / 2 / self.tlen)))
        return base_wave


class CosineEdgeRect(Envelope):
    def __init__(self, t0, tlen, amp, w):
        self.tlen = tlen
        self.w = w
        super().__init__(start=t0 - self.w,
                         end=t0 + tlen + self.w,
                         amp=amp,
                         cache_characteristic=(round(self.w / TIME_ATOL),
                                               round(self.tlen / TIME_ATOL)))

    def model(self, base_start: float, base_t: np.ndarray):
        base_t0 = base_start + self.w
        base_tmin = base_t0
        base_tmax = base_t0 + self.tlen
        base_wave = 1.0 * (base_t >= base_tmin) * (base_t < base_tmax)
        base_wave += np.sin(0.5 * np.pi * (base_t - base_start) / self.w) * (
            base_t >= base_start) * (base_t < base_start + self.w)
        base_wave += np.cos(0.5 * np.pi * (base_t - base_tmax) / self.w) * (
            base_t >= base_tmax) * (base_t < base_tmax + self.w)
        return base_wave


class EllipseEdgeRect(Envelope):
    def __init__(self, t0, tlen, amp, w):
        self.tlen = tlen
        self.w = w
        super().__init__(start=t0 - self.w,
                         end=t0 + tlen + self.w,
                         amp=amp,
                         cache_characteristic=(round(self.w / TIME_ATOL),
                                               round(self.tlen / TIME_ATOL)))

    def model(self, base_start: float, base_t: np.ndarray):
        base_t0 = base_start + self.w
        base_tmin = base_t0
        base_tmax = base_t0 + self.tlen
        base_wave = 1.0 * (base_t >= base_tmin) * (base_t < base_tmax)
        base_wave += np.sqrt(
            (1 - ((base_t - base_tmin) / self.w)**2) * (base_t >= base_start) *
            (base_t < base_start + self.w))
        base_wave += np.sqrt(
            (1 - ((-base_t + base_tmax) / self.w)**2) * (base_t >= base_tmax) *
            (base_t < base_tmax + self.w))
        return base_wave


# ------- align -----


def align(env: AbstractEnvelope,
          dt: np.ndarray,
          amp: Optional[np.ndarray] = None) -> EnvSum:
    if isinstance(env, EnvSum):
        result = 0
        for e in env.items:
            result += align(e, dt, amp)
        return result
    if amp is None:
        result = EnvSum()
        result.items = [(env >> dt_) for dt_ in dt]
        return result
    else:
        result = EnvSum()
        result.items = [(env >> dt_) * amp_ for dt_, amp_ in zip(dt, amp)]
        return result


def split(env: AbstractEnvelope, starts: List[float],
          ends: List[float]) -> List[AbstractEnvelope]:
    """
    Split env according to starts and ends.
    The provided starts and ends must be pre-sorted and have the same length.
    """
    splitted_envs = []
    if len(starts) == 1:
        if env.start < starts[0] or ends[0] < env.end:
            raise RuntimeError(
                f'env:start={env.start} end={env.end}\nblock:start={starts[0]} end={ends[0]}\nblock can not cover the envelope!'
            )
        splitted_envs.append(env)
    else:
        splitted_envs = [EnvSum() for _ in range(len(starts))]
        if isinstance(env, EnvSum):
            for _item in env.items:
                found = False
                for splitted_env, start, end in zip(splitted_envs, starts,
                                                    ends):
                    if _item.start >= start and _item.end <= end:
                        splitted_env += _item
                        found = True
                    elif (_item.start < start <
                          _item.end) or (_item.start < end < _item.end):
                        raise RuntimeError(
                            f'env:start={_item.start} end={_item.end}\nblock:start={start} end={end}\nblock can not cover the envelope!'
                        )
                if not found:
                    raise RuntimeError("There is env not in any block!")
        else:
            found = False
            for splitted_env, start, end in zip(splitted_envs, starts, ends):
                if env.start >= start and env.end <= end:
                    splitted_env += env
                    found = True
                elif (env.start < start < env.end) or (env.start < end <
                                                       env.end):
                    raise RuntimeError(
                        f'env:start={env.start} end={env.end}\nblock:start={start} end={end}\nblock can not cover the envelope!'
                    )
            if not found:
                raise RuntimeError("There is env not in any block!")
        is_complex = env.is_complex
        for idx, (splitted_env, start) in enumerate(zip(splitted_envs,
                                                        starts)):
            if len(splitted_env.items) == 0:
                splitted_env += Rect(start, 0.0, 0.0)
            if is_complex and not splitted_env.is_complex:
                splitted_envs[idx] = splitted_env * (1 + 0j)
    return splitted_envs


# ------- decode envelope -----


def decode_envelope(
        env: AbstractEnvelope,
        wc: Optional[Union[None, WaveCache]] = None,
        start: Optional[Union[None, float]] = None,
        end: Optional[Union[None, float]] = None) -> Tuple[float, np.ndarray]:
    '''
    Decode an envelope into a sequence of (t, wave) pairs.
    Resoluton of the sequence is determined by wc (default is 0.5).
    If start and end are not specified, the envelope's start and end are used.
    If wc is not specified, a new one is created.
    '''
    if wc is None:
        wc = WaveCache()
    t_start, wave = env.time_func(wc)
    if isinstance(wave, LazyArray):
        wave = wave.eval()
    t_end = t_start + (len(wave) - 1) * wc.resolution
    if start is not None:
        start = round(
            start / TIME_ATOL) // wc.resolution_over_atol * wc.resolution
        if start < t_start:
            prepad_sample_num = round((t_start - start) / wc.resolution)
            wave = np.concatenate((np.zeros(prepad_sample_num,
                                            dtype=wave.dtype), wave))
        else:
            assert start < t_end
            start_idx = round((start - t_start) / wc.resolution)
            wave = wave[start_idx:]
    else:
        start = t_start
    if end is not None:
        end = (round(end / TIME_ATOL) // wc.resolution_over_atol -
               1) * wc.resolution
        if end > t_end:
            postpad_sample_num = round((end - t_end) / wc.resolution)
            wave = np.concatenate(
                (wave, np.zeros(postpad_sample_num, dtype=wave.dtype)))
        else:
            assert end > start
            end_idx = round((t_end - end) / wc.resolution)
            wave = wave[:-end_idx]
    else:
        end = t_end
    return start, wave
