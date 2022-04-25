"""
Prepare a langdir and dictionary given a corpus
"""

import argparse
from pathlib import Path
import shutil
from typing import List

from uk.g2p import g2p
from uk.subprocess import sh, check_output


def extend_dict(words: List[str], dict_dir: Path, source_dict_dir: Path):
    shutil.copy(source_dict_dir / 'extra_questions.txt', dict_dir / 'extra_questions.txt')
    shutil.copy(source_dict_dir / 'optional_silence.txt', dict_dir / 'optional_silence.txt')
    shutil.copy(source_dict_dir / 'silence_phones.txt', dict_dir / 'silence_phones.txt')
    shutil.copy(source_dict_dir / 'nonsilence_phones.txt', dict_dir / 'nonsilence_phones.txt')

    with open(dict_dir / 'lexicon.txt', 'w') as f:
        with open(source_dict_dir / 'lexicon.txt') as lexicon:
            oov = {}
            for word in words:
                oov[word] = ' '.join(g2p(word))

            for line in lexicon:
                word, prons = line.split(maxsplit=1)
                print(word, prons.strip(), file=f)
                if word in oov:
                    del oov[word]

        for word in oov:
            print(word, oov[word].strip(), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""\
    Prepare a langdir and dictionary given a corpus

    python3 -m uk.prepare_lang -w exp/lang -o data/lang data/local/semesyuk_farshrutka/01_prologue.txt
    """)
    parser.add_argument('-o', '--output-dir', type=Path, required=True)
    parser.add_argument('-d', '--dict-dir', type=Path, default='data/local/dict', help='source dictionary')
    parser.add_argument('corpus_txt', type=Path, help='must be tokenized (see README)')

    args = parser.parse_args()

    dict_dir = args.output_dir / 'dict'
    dict_dir.mkdir(exist_ok=True, parents=True)

    langdir = args.output_dir / 'lang'

    words = args.corpus_txt.read_text().split()
    extend_dict(words, dict_dir, args.dict_dir)

    sh('utils/prepare_lang.sh', dict_dir, "<unk>", args.output_dir / 'lang_tmp', langdir)
