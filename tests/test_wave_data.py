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


def test_evc_split():
    wc = evc.WaveCache(0.5)
    g = evc.Gaussian(0, 10, 0.5)
    shifts = np.array([100.0, 200.0])
    evl = evc.align(g, shifts)
    start = evl.start()
    end = evl.end()
    spltted_evl = evc.split(evl, shifts - 50, shifts + 50)
    assert np.allclose(np.array(evc.decode_envelope(g >> shifts[0], wc, start,
                                                    end)[1],
                                copy=False),
                       np.array(evc.decode_envelope(spltted_evl[0], wc, start,
                                                    end)[1],
                                copy=False),
                       atol=ATOL)
    assert np.allclose(np.array(evc.decode_envelope(g >> shifts[1], wc, start,
                                                    end)[1],
                                copy=False),
                       np.array(evc.decode_envelope(spltted_evl[1], wc, start,
                                                    end)[1],
                                copy=False),
                       atol=ATOL)

    evl = g >> 100
    spltted_evl = evc.split(evl, shifts - 50, shifts + 50)
    assert np.allclose(np.array(evc.decode_envelope(g >> shifts[0], wc, start,
                                                    end)[1],
                                copy=False),
                       np.array(evc.decode_envelope(spltted_evl[0], wc, start,
                                                    end)[1],
                                copy=False),
                       atol=ATOL)
    wave = np.array(evc.decode_envelope(spltted_evl[1], wc, start, end)[1],
                    copy=False)
    assert np.allclose(wave, np.zeros_like(wave), atol=ATOL)

    try:
        evc.split(evl, shifts + 1000, shifts + 1050)
        raise ValueError("Last line should raise RuntimeError")
    except RuntimeError:
        pass

    # If original envelope is complex, the splitted envelopes need to be complex as well.
    evl = evc.Gaussian(shifts[0], 10, 0.5) + evc.GaussianDRAG(
        shifts[1], 10, 0.2, 0.1, 0.1, 0.0)
    spltted_evl = evc.split(evl, shifts - 50, shifts + 50)
    for _evl in spltted_evl:
        assert _evl.is_complex()

    evl = evc.Gaussian(shifts[0], 10, 0.5) + evc.Gaussian(shifts[1], 10, 0.2)
    spltted_evl = evc.split(evl, shifts - 50, shifts + 50)
    for _evl in spltted_evl:
        assert not _evl.is_complex()


def test_evp_split():
    wc = evp.WaveCache(0.5)
    g = evp.Gaussian(0, 10, 0.5)
    shifts = np.array([100.0, 200.0])
    evl = evp.align(g, shifts)
    start = evl.start
    end = evl.end
    spltted_evl = evp.split(evl, shifts - 50, shifts + 50)
    assert np.allclose(np.array(evp.decode_envelope(g >> shifts[0], wc, start,
                                                    end)[1],
                                copy=False),
                       np.array(evp.decode_envelope(spltted_evl[0], wc, start,
                                                    end)[1],
                                copy=False),
                       atol=ATOL)
    assert np.allclose(np.array(evp.decode_envelope(g >> shifts[1], wc, start,
                                                    end)[1],
                                copy=False),
                       np.array(evp.decode_envelope(spltted_evl[1], wc, start,
                                                    end)[1],
                                copy=False),
                       atol=ATOL)

    evl = g >> 100
    spltted_evl = evp.split(evl, shifts - 50, shifts + 50)
    assert np.allclose(np.array(evp.decode_envelope(g >> shifts[0], wc, start,
                                                    end)[1],
                                copy=False),
                       np.array(evp.decode_envelope(spltted_evl[0], wc, start,
                                                    end)[1],
                                copy=False),
                       atol=ATOL)
    wave = np.array(evp.decode_envelope(spltted_evl[1], wc, start, end)[1],
                    copy=False)
    assert np.allclose(wave, np.zeros_like(wave), atol=ATOL)

    try:
        evp.split(evl, shifts + 1000, shifts + 1050)
        raise ValueError("Last line should raise RuntimeError")
    except RuntimeError:
        pass

    evl = evp.Gaussian(shifts[0], 10, 0.5) + evp.GaussianDRAG(
        shifts[1], 10, 0.2, 0.1, 0.1, 0.0)
    spltted_evl = evp.split(evl, shifts - 50, shifts + 50)
    for _evl in spltted_evl:
        assert _evl.is_complex

    evl = evp.Gaussian(shifts[0], 10, 0.5) + evp.Gaussian(shifts[1], 10, 0.2)
    spltted_evl = evp.split(evl, shifts - 50, shifts + 50)
    for _evl in spltted_evl:
        assert not _evl.is_complex
