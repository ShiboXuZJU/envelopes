import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
print(sys.path[-1])
from envelopes import evc, evp
import numpy as np
import math

ATOL = 1e-8


def test_evc():
    t0 = 5
    w = 10
    amp = 0.1
    shift = 100
    resolution = 0.5
    wc_c = evc.WaveCache(resolution)

    g = evc.Gaussian(t0=t0, w=w, amp=amp)
    sigma = w / math.sqrt(8 * math.log(2))
    t_, wave = evc.decode_envelope(g, wc_c)
    wave = np.array(wave, copy=False)
    t = np.arange(t_, t_ + len(wave) * resolution, resolution)
    assert np.allclose(wave,
                       amp * np.exp(-(t - t0)**2 / 2 / sigma**2),
                       atol=ATOL)

    t_, wave = evc.decode_envelope(g, wc_c)
    wave = np.array(wave, copy=False)
    t = np.arange(t_, t_ + len(wave) * resolution, resolution)
    assert np.allclose(wave,
                       amp * np.exp(-(t - t0)**2 / 2 / sigma**2),
                       atol=ATOL)

    t_, wave = evc.decode_envelope(2 * g, wc_c)
    wave = np.array(wave, copy=False)
    t = np.arange(t_, t_ + len(wave) * resolution, resolution)
    wave_num = 2 * amp * np.exp(-(t - t0)**2 / 2 / sigma**2)
    assert np.allclose(wave, wave_num, atol=ATOL)

    t_, wave = evc.decode_envelope(2 * g + (g >> shift), wc_c)
    wave = np.array(wave, copy=False)
    t = np.arange(t_, t_ + len(wave) * resolution, resolution)
    wave_num = 2 * amp * np.exp(-(t - t0)**2 / 2 / sigma**2) + amp * np.exp(
        -(t - (t0 + shift))**2 / 2 / sigma**2)
    assert np.allclose(wave, wave_num, atol=ATOL)

    df = 0.1323
    phase = 0.12
    t_, wave = evc.decode_envelope(
        evc.mix(g, df=df, phase=phase, dynamical=True), wc_c)
    wave = np.array(wave, copy=False)
    t = np.arange(t_, t_ + len(wave) * resolution, resolution)
    assert np.allclose(wave,
                       amp * np.exp(-(t - t0)**2 / 2 / sigma**2) *
                       np.exp(-2j * np.pi * df * t + 1j * phase),
                       atol=ATOL)
    # test dynamical mixing
    t_, wave = evc.decode_envelope(
        evc.mix(g, df=df, phase=phase, dynamical=True) >> shift, wc_c)
    wave = np.array(wave, copy=False)
    t = np.arange(t_, t_ + len(wave) * resolution, resolution)
    assert np.allclose(wave,
                       amp * np.exp(-(t - (t0 + shift))**2 / 2 / sigma**2) *
                       np.exp(-2j * np.pi * df * t + 1j * phase),
                       atol=ATOL)
    t_, wave = evc.decode_envelope(
        evc.mix(g >> shift, df=df, phase=phase, dynamical=True), wc_c)
    wave = np.array(wave, copy=False)
    t = np.arange(t_, t_ + len(wave) * resolution, resolution)
    assert np.allclose(wave,
                       amp * np.exp(-(t - (t0 + shift))**2 / 2 / sigma**2) *
                       np.exp(-2j * np.pi * df * t + 1j * phase),
                       atol=ATOL)
    # test static mixing
    t_, wave = evc.decode_envelope(
        evc.mix(g, df=df, phase=phase, dynamical=False) >> shift, wc_c)
    wave = np.array(wave, copy=False)
    t = np.arange(t_, t_ + len(wave) * resolution, resolution)
    assert np.allclose(wave,
                       amp * np.exp(-(t - (t0 + shift))**2 / 2 / sigma**2) *
                       np.exp(-2j * np.pi * df * (t - shift) + 1j * phase),
                       atol=ATOL)
    t_, wave = evc.decode_envelope(
        evc.mix(g >> shift, df=df, phase=phase, dynamical=False), wc_c)
    wave = np.array(wave, copy=False)
    t = np.arange(t_, t_ + len(wave) * resolution, resolution)
    assert np.allclose(wave,
                       amp * np.exp(-(t - (t0 + shift))**2 / 2 / sigma**2) *
                       np.exp(-2j * np.pi * df * t + 1j * phase),
                       atol=ATOL)


def test_evc_decode_envelope_with_start_end():
    t0 = 5
    w = 10
    amp = 0.1
    resolution = 0.5
    wc_c = evc.WaveCache(resolution)

    g = evc.Gaussian(t0=t0, w=w, amp=amp)
    sigma = w / math.sqrt(8 * math.log(2))

    t_, wave = evc.decode_envelope(g, wc_c)
    wave = np.array(wave, copy=False)
    t = np.arange(t_, t_ + len(wave) * resolution, resolution)
    assert np.allclose(wave,
                       amp * np.exp(-(t - t0)**2 / 2 / sigma**2),
                       atol=ATOL)

    t_, wave = evc.decode_envelope(g, wc_c, start=2, end=5)
    wave = np.array(wave, copy=False)
    t = np.arange(t_, t_ + len(wave) * resolution, resolution)
    assert np.allclose(wave,
                       amp * np.exp(-(t - t0)**2 / 2 / sigma**2),
                       atol=ATOL)

    t_, wave = evc.decode_envelope(g, wc_c, start=-100, end=100)
    wave = np.array(wave, copy=False)
    t = np.arange(t_, t_ + len(wave) * resolution, resolution)
    assert np.allclose(wave,
                       amp * np.exp(-(t - t0)**2 / 2 / sigma**2),
                       atol=ATOL)
