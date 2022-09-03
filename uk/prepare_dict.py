from collections import defaultdict
from pathlib import Path

from .subprocess import sh


def prepare_dict(mfa_lexicon: Path, dict_dir: Path):
    with open(dict_dir / 'silence_phones.txt', 'w') as f:
        print('sil', file=f)

    with open(dict_dir / 'optional_silence.txt', 'w') as f:
        print('sil', file=f)

    # An extra question will be added by including the silence phones in one class.
    with open(dict_dir / 'extra_questions.txt', 'w') as f:
        print('sil', file=f)

    syms = set()
    lexicon = defaultdict(list)
    with open(mfa_lexicon) as f:
        for line in f:
            if '\t' in line:
                # assume MFA format
                word, p1, p2, p3, p4, pron = line.strip().split('\t')
            else:
                word, prons = line.strip().split()
                pron = ' '.join(prons)
            lexicon[word].append(pron)
            for sym in pron.split():
                syms.add(sym)

    with open(dict_dir / 'nonsilence_phones.txt', 'w') as f:
        for sym in sorted(syms):
            print(sym, file=f)

    lexicon['<unk>'].append('sil')

    with open(dict_dir / 'lexicon.txt', 'w') as f:
        for word in sorted(lexicon):
            for pron in lexicon[word]:
                print(word, pron, file=f)

    sh('utils/validate_dict_dir.pl', dict_dir)
    sh('utils/lang/make_unk_lm.sh', dict_dir, 'exp/make_unk')


if __name__ == '__main__':
    dict_dir = Path('data/english')
    dict_dir.mkdir(exist_ok=True)
    prepare_dict(Path('data/local/dict/english_mfa_reference.dict'), dict_dir)