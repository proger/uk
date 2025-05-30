"""
Prepare a langdir and dictionary given a corpus
"""

import argparse
from pathlib import Path
import shutil
from typing import List

from loguru import logger

from uk.g2p import G2PBatched, g2p_batch
from uk.subprocess import sh


def extend_dict(words: List[str], dict_dir: Path, source_dict_dir: Path, g2p_batch: G2PBatched = g2p_batch) -> None:
    extra_questions = 'extra_questions.txt'
    if (source_dict_dir / extra_questions).exists():
        shutil.copy(source_dict_dir / extra_questions, dict_dir / extra_questions)
    shutil.copy(source_dict_dir / 'optional_silence.txt', dict_dir / 'optional_silence.txt')
    shutil.copy(source_dict_dir / 'silence_phones.txt', dict_dir / 'silence_phones.txt')
    shutil.copy(source_dict_dir / 'nonsilence_phones.txt', dict_dir / 'nonsilence_phones.txt')

    oov = set(words)

    with open(dict_dir / 'lexicon.txt', 'w') as f:
        with open(source_dict_dir / 'lexicon.txt') as lexicon:
            for line in lexicon:
                word, prons = line.split(maxsplit=1)
                print(word, prons.strip(), file=f)
                if word in oov:
                    oov.remove(word)

        new_lexicon = dict(g2p_batch(sorted(oov)))

        for word in new_lexicon:
            if len(word) > 20:
                logger.warning('word too long: {}', word)
            print(word, new_lexicon[word].strip(), file=f)


def format_ngram_lm(corpus_txt: Path, work_dir: Path, lexiconp_txt: Path, order=2):
    sh('ngram-count', '-text', corpus_txt,
                      '-order', order, '-unk', '-lm',
                       work_dir / "arpa")
    sh('gzip', '-f', work_dir / 'arpa')
    sh('utils/format_lm.sh', work_dir / 'lang', work_dir / 'arpa.gz', lexiconp_txt, work_dir / 'lang')


if __name__ == '__main__':
    from uk.dynamic import import_function

    parser = argparse.ArgumentParser(description="""\
    Prepare a langdir, dictionary and LM given a corpus

    python3 -m uk.prepare_lang -w exp/lang -o data/lang data/local/semesyuk_farshrutka/01_prologue.txt
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output-dir', type=Path, required=True)
    parser.add_argument('-d', '--dict-dir', type=Path, default='exp/dict', help='source dictionary')
    parser.add_argument('--g2p_batch', type=import_function, default='uk.g2p:g2p_batch', help='batched g2p implementation')
    parser.add_argument('--format-lm', action='store_true', help='compute LM from corpus using ngram-count and run utils/format_lm.sh')
    parser.add_argument('--text', action='store_true', default='corpus_txt is kaldi text')
    parser.add_argument('corpus_txt', type=Path, help='must be tokenized (see README)')

    args = parser.parse_args()

    dict_dir = args.output_dir / 'dict'
    dict_dir.mkdir(exist_ok=True, parents=True)

    langdir = args.output_dir / 'lang'

    if args.text:
        words = []
        with open(args.corpus_txt) as f:
            for line in f:
                words.extend(line.strip().split()[1:])
    else:
        words = args.corpus_txt.read_text().split()
    extend_dict(words, dict_dir, args.dict_dir)

    sh('utils/prepare_lang.sh', dict_dir, "<unk>", args.output_dir / 'lang_tmp', langdir)
    if args.format_lm:
        format_ngram_lm(args.corpus_txt, args.output_dir, args.output_dir / 'lang_tmp/lexiconp.txt', order=3)
