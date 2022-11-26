from collections import defaultdict
from pathlib import Path
import re

from loguru import logger

from .subprocess import sh


def sym_key_(sym, _digits=re.compile(r'[0-9]')):
    # all phones are the same up to some number (needs extra_questions to be adjusted)
    return _digits.sub('', sym)

def sym_key(sym):
    return sym


def read_lexicon(lexicon_path: Path):
    syms = defaultdict(set)
    lexicon = defaultdict(list)
    has_mfa = False
    with open(lexicon_path) as f:
        for line in f:
            if '\t' in line:
                # assume MFA format
                if not has_mfa:
                    logger.info('assuming MFA lexicon format for {}', lexicon_path)
                    has_mfa = True
                word, p1, p2, p3, p4, pron = line.strip().split('\t')
            else:
                word, *prons = line.strip().split()
                pron = ' '.join(prons)
            lexicon[word].append(pron)
            for sym in pron.split():
                syms[sym_key(sym)].add(sym)
    return syms, lexicon


def prepare_dict(lexicon: Path, dict_dir: Path, make_unk=True):
    dict_dir.mkdir(exist_ok=True, parents=True)

    with open(dict_dir / 'silence_phones.txt', 'w') as f:
        print('sil', file=f)

    with open(dict_dir / 'optional_silence.txt', 'w') as f:
        print('sil', file=f)

    # An extra question will be added by including the silence phones in one class.
    with open(dict_dir / 'extra_questions.txt', 'w') as f:
        print('sil', file=f)

    syms, lexicon = read_lexicon(lexicon)

    with open(dict_dir / 'nonsilence_phones.txt', 'w') as f:
        for sym in sorted(syms):
            print(*sorted(syms[sym]), file=f)

    lexicon['<unk>'].append('sil')

    with open(dict_dir / 'lexicon.txt', 'w') as f:
        for word in sorted(lexicon):
            for pron in lexicon[word]:
                print(word, pron, file=f)

    sh('utils/validate_dict_dir.pl', dict_dir)

    if make_unk:
        sh('utils/lang/make_unk_lm.sh', dict_dir, dict_dir / 'unk')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""\
    Prepare dictionary

    python3 -m uk.prepare_dict -o data/english data/local/dict/english_mfa_reference.dict
    python3 -m uk.prepare_dict -o data/local/dict data/local/dict/lexicon_common_voice_uk.txt
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output-dict-dir', type=Path, default='exp/dict', help='output dictionary')
    parser.add_argument('lexicon', type=Path, help='lexicon file (letters to sounds, mfa dict or kaldi lexicon.txt)')
    args = parser.parse_args()

    prepare_dict(args.lexicon, args.output_dict_dir, make_unk=False)