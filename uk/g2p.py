"""
Transduce graphemes to phonemes
"""

import os
import warnings
from typing import Sequence, Mapping, Callable

warnings.warn('Setting CUDA_VISIBLE_DEVICES="" due to bug in ukro-g2p==0.1.5')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from ukro_g2p.predict import G2P

__all__ = ['g2p', 'g2p_batch', 'G2PBatched']

G2PBatched = Callable[[Sequence[str]], Mapping[str, str]]

g2p_base = G2P('ukro-base-uncased')

replacements = {
    'SH23': 'SH2',
    'H3': 'H',
    'M3': 'M',
    'RJ3': 'RJ',
}


def g2p(word):
    return [replacements.get(p, p) for p in g2p_base(word)]


def g2p_batch(words: Sequence[str]) -> Mapping[str, str]:
    oov = {}
    for word in words:
        oov[word] = ' '.join(g2p(word))
    return oov