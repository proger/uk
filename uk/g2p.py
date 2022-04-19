"""
Transduce graphemes to phonemes
"""

import os
import warnings

warnings.warn('Setting CUDA_VISIBLE_DEVICES="" due to bug in ukro-g2p==0.1.5')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from ukro_g2p.predict import G2P

__all__ = ['g2p']

g2p_base = G2P('ukro-base-uncased')

replacements = {
    'SH23': 'SH2',
    'H3': 'H',
}

def g2p(word):
    return [replacements.get(p, p) for p in g2p_base(word)]