"""
Prepare Common Voice dataset for training using Kaldi
"""

from collections import defaultdict
from contextlib import ExitStack
import os
from pathlib import Path
import re
from typing import Dict, Set

import datasets
from loguru import logger
from sqlite_utils import Database
import torch
import torchaudio
from tqdm import tqdm


def keep_useful_characters(s, alphabet='cyr'):
    s = s.lower()
    s = s.replace('’', "'")
    if alphabet == 'latin':
        s = re.sub(r'[^A-Za-z\' -]', '', s)
    elif alphabet == 'cyr':
        s = re.sub(r'[^ыёэъЫЁЭЪйцукенгшщзхїфивапролджєґячсміiтьбюЙЦУКЕНГШЩЗХЇФИВАПРОЛДЖЄҐЯЧСМІТЬБЮ\' -]', '', s)
    else:
        s =         re.sub(r'[^йцукенгшщзхїфивапролджєґячсміiтьбюЙЦУКЕНГШЩЗХЇФИВАПРОЛДЖЄҐЯЧСМІТЬБЮ\' -]', '', s)
    s = re.sub(r'[ -]+', ' ', s)
    s = re.sub(r'\s+', ' ', s) # unicode whitespace
    s = re.sub(r'^\'+', '', s) # leading apostrophes
    s = re.sub(r'\'+$', '', s) # trailing apostrophes
    if alphabet != 'latin':
        s = s.replace('i','і')
    s = s.strip()
    return s


def tokens(s):
    return s.split()


def write_scp(scp: Dict[str, str], filename: Path):
    with open(filename, 'w') as f:
        for utterance_id in sorted(scp):
            print(utterance_id, scp[utterance_id], file=f)


def write_spk2utt(spk2utt: Dict[str, Set[str]], datadir: Path):
    with open(datadir / 'spk2utt', 'w') as f:
        for speaker_id in sorted(spk2utt):
            print(speaker_id, end='', file=f)
            for utterance_id in sorted(spk2utt[speaker_id]):
                print('', utterance_id, end='', file=f)
            print(file=f)


def prepare(dataset, datadir, g2p=None, alphabet='cyr', copy_wav=False):
    datadir.mkdir(exist_ok=True, parents=True)
    (datadir / 'wav').mkdir(exist_ok=True)

    db = Database(datadir / 'db.sqlite', recreate=True)

    text = {}
    utt2spk = {}
    spk2utt = defaultdict(set)
    wavscp = {}

    with ExitStack() as stack:
        lexicon = {}
        words_txt = stack.enter_context(open(datadir / 'words.txt', 'w'))
        if g2p is not None:
            lexicon_txt = stack.enter_context(open(datadir / 'lexicon.txt', 'w'))

        for sample in tqdm(dataset):
            utterance_id = str(sample.get('id') or Path(sample['path']).stem)
            speaker_id = str(sample.get('speaker_id', utterance_id))

            orig_sentence = sample.get('sentence') or sample['text']
            sentence = keep_useful_characters(orig_sentence, alphabet=alphabet)
            words = [keep_useful_characters(t, alphabet=alphabet) for t in tokens(sentence)]

            text[utterance_id] = ' '.join(words)
            utt2spk[utterance_id] = speaker_id
            spk2utt[speaker_id].add(utterance_id)

            if copy_wav or not 'path' in sample['audio']:
                sampling_rate = sample['audio'].get('sampling_rate', 48000)
                loc = (datadir / 'wav' / utterance_id).with_suffix('.wav')
                waveform = torch.from_numpy(sample['audio']['array'])[None, :].float()
                torchaudio.save(loc, waveform, sampling_rate, bits_per_sample=16, encoding='PCM_S')
                wavscp[utterance_id] = f'sox {loc} -r 16k -t wav -c 1 - |'
            else:
                loc = sample['audio']['path']
                wavscp[utterance_id] = f'sox {loc} -r 16k -t wav -c 1 - |'

            db['utterances'].insert(dict(utterance_id=utterance_id,
                                         text=text[utterance_id],
                                         orig_text=orig_sentence,
                                         spk=utt2spk[utterance_id],
                                         media=loc), pk='utterance_id')

            for word in words:
                if not word in lexicon:
                    lexicon[word] = None

        if g2p is not None:
            for word in lexicon:
                if pron := g2p(word):
                    lexicon[word] = ' '.join(pron)

        for word in sorted(lexicon):
            if lexicon[word]:
                print(word, lexicon[word], file=lexicon_txt)
            print(word, file=words_txt)

    db['utterances'].enable_fts(['text', 'orig_text'])

    write_scp(text, datadir / 'text')
    write_scp(utt2spk, datadir / 'utt2spk')
    write_spk2utt(spk2utt, datadir)
    write_scp(wavscp, datadir / 'wav.scp')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(__file__, description='prepare kaldi data directory with a speech dataset from Hugging Face',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lexicon', action='store_true',
                        help='generate lexicon for every word using ukro-g2p')
    parser.add_argument('--dataset', default='mozilla-foundation/common_voice_10_0',
                        help='dataset name on Hugging Face')
    parser.add_argument('--subset', default='uk',
                        help='subset (language code for Common Voice)')
    parser.add_argument('--alphabet', default='uk', choices=('uk', 'latin', 'cyr'),
                        help='alphabet to use for keep_useful_characters')
    parser.add_argument('--root', type=Path, default=Path('data'),
                        help='where to put {lang}/test and {lang}/train datadirs')
    parser.add_argument('--split', type=str, default='train',
                        help='split to generate (train, validation, test, etc)')

    args = parser.parse_args()

    logger.info('{}', args)

    datadir = args.root / args.dataset / args.subset / args.split
    logger.info('writing to {}', datadir)

    uk = datasets.load_dataset(args.dataset, name=args.subset, split=args.split, use_auth_token=True)

    if args.lexicon:
        from uk.g2p import g2p
    else:
        g2p = None

    prepare(uk, datadir, g2p=g2p, alphabet=args.alphabet)
