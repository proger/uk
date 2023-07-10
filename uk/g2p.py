"""
Transduce graphemes to phonemes
"""

import os
import warnings
from pathlib import Path
from typing import Sequence, Mapping, Callable

from loguru import logger

warnings.warn('Setting CUDA_VISIBLE_DEVICES="" due to bug in ukro-g2p==0.1.5')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from ukro_g2p.predict import G2P

from uk.prepare_dict import read_lexicon

__all__ = ['g2p', 'g2p_batch', 'G2PBatched']

G2PBatched = Callable[[Sequence[str]], Mapping[str, str]]

g2p_base = G2P('ukro-base-uncased')

# X3 means prolonged version of X
replacements = {
    'SH23': 'SH2',
    'H3': 'H',
    'M3': 'M',
    'RJ3': 'RJ',
    'F3': 'F', # баффало
    'B23': 'B2', # оббігав
    'X3': 'X', # ваххабіт
    'K3': 'K',
    'P3': 'P',
    'TS3': 'TS',
    'Y3': 'Y',
    'B3': 'B',
    'ZJ3': 'ZJ',
    'ZH3': 'ZH',
    'P23': 'P2', # філіппінський
}

_, reference_lexicon = read_lexicon(Path(__file__).parent / '../data/local/dict/lexicon_common_voice_uk.txt')


def g2p(word):
    pron = reference_lexicon.get(word)
    if not pron:
        pron = [replacements.get(p, p) for p in g2p_base(word)]
    return pron


def g2p_batch(words: Sequence[str]) -> Mapping[str, str]:
    oov = {}
    for word in words:
        try:
            word = word.replace('i', 'і') # oops
            oov[word] = ' '.join(g2p(word))
        except:
            logger.warning('failed on word: {}', word)
    return oov
