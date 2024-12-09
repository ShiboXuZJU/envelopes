from . import envelopes_cpp as evc


def DRAG(start: float,
         amp: float,
         length: float,
         alpha: float,
         nonlinearity: float,
         mixing_freq: float,
         phase: float = 0.0,
         profile: str = 'gaussian') -> evc.AbstractEnvelope:
    '''
    Implements the DRAG.
    ------
    nonlinearity [GHz] = f21-f10
    '''
    if profile == 'gaussian':
        w = length / 2
        evl = evc.GaussianDRAG(t0=start + length / 2,
                               w=w,
                               coef=-alpha / nonlinearity,
                               df=mixing_freq,
                               amp=amp,
                               phase=phase)
    elif profile == 'cosine':
        w = length
        evl = evc.CosineDRAG(t0=start + length / 2,
                             w=w,
                             coef=-alpha / nonlinearity,
                             df=mixing_freq,
                             amp=amp,
                             phase=phase)
    else:
        raise ValueError('Invalid profile')
    return evl
