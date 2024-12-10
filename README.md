# envelopes

High performance tools for building and decoding envelopes used in superconducting quantum qubit experiments.

# Installation
```bash
pip install envelopes-qc
```
 Or
1. Cloning the repository to your machine.
2. Installing envelopes using pip locally by the following command:
```bash
python setup.py build_ext --inplace
```
or directly building package under the folder "site-package":
```bash
pip install .
```

# Usage
```python
from envelopes import evc, DRAG # evc (evp) is the envelopes module implemented by C++ (Python)
import numpy as np
# create an envelope
pi_pulse = DRAG(start=0, amp=1, length=30, alpha=0.5, nonlinearity=-0.2, mixing_freq=0.1, phase=0.0, profile='gaussian') # profile can be 'gaussian' or 'cosine'
amp = np.array([1, 0.3, 0.7, 0.8])
dt = np.array([0, 50, 100, 200])
xy = evc.align(pi_pulse, dt, amp) # align the envelope to the given dt and amplitude
# The following code has equivalent effect but undermine the efficiency:
# xy = 0
# for _dt, _amp in zip(dt, amp):
#     xy += (pi_pulse >> dt) * amp

# decode the envelope
resolution = 0.5
wc = evc.WaveCache(resolution)
# t, wave = evc.decode_envelope(xy, wc) default start and end defined by the envelope will be used
t_start, wave = evc.decode_envelope(xy, wc, start=-50, end=250)
wave = np.array(wave, copy=False) # convert vector wrapper to numpy array
t_list = np.arange(t_start, t_start+len(wave)*resolution, resolution)


import matplotlib.pyplot as plt
plt.figure()
plt.plot(t_list, wave.real, '.-', label='real')
plt.plot(t_list, wave.imag, '.-', label='imag')
plt.legend()
plt.xlabel('time')
plt.ylabel('amplitude')
plt.tight_layout()
plt.show()
```

## Supported Envelopes
See [tests/supported_envelopes.ipynb](./tests/supported_envelopes.ipynb).

## Speed test
See [tests/speed_test.ipynb](./tests/speed_test.ipynb).