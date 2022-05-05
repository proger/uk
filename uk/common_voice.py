"""
Prepare Common Voice dataset for training using Kaldi
"""

from contextlib import ExitStack
import os
from pathlib import Path
import re

import datasets
import torch
import torchaudio
from tqdm import tqdm


def keep_useful_characters(s, alphabet='cyr'):
    s = s.lower()
    s = s.replace('’', "'")
    if alphabet == 'cyr':
        s = re.sub(r'[^ыёэъЫЁЭЪйцукенгшщзхїфивапролджєґячсміiтьбюЙЦУКЕНГШЩЗХЇФИВАПРОЛДЖЄҐЯЧСМІТЬБЮ\' -]', '', s)
    else:
        s = re.sub(r'[^йцукенгшщзхїфивапролджєґячсміiтьбюЙЦУКЕНГШЩЗХЇФИВАПРОЛДЖЄҐЯЧСМІТЬБЮ\' -]', '', s)
    s = re.sub(r'[ -]+', ' ', s)
    s = re.sub(r'\s+', ' ', s) # unicode whitespace
    s = s.replace('i','і')
    s = s.strip()
    return s


def prepare(dataset, datadir, g2p=None):
    datadir.mkdir(exist_ok=True, parents=True)
    (datadir / 'wav').mkdir(exist_ok=True)

    with ExitStack() as stack:
        text = stack.enter_context(open(datadir / 'text', 'w'))
        utt2spk = stack.enter_context(open(datadir / 'utt2spk', 'w'))
        spk2utt = stack.enter_context(open(datadir / 'spk2utt', 'w'))
        wavscp = stack.enter_context(open(datadir / 'wav.scp', 'w'))

        lexicon = {}
        words_txt = stack.enter_context(open(datadir / 'words.txt', 'w'))
        if g2p is not None:
            lexicon_txt = stack.enter_context(open(datadir / 'lexicon.txt', 'w'))

        for sample in tqdm(dataset):
            path = Path(sample['path'])
            loc = (datadir / 'wav' / path.name).with_suffix('.wav')
            sentence = keep_useful_characters(sample['sentence'])
            print(path.stem, sentence, file=text)
            print(path.stem, path.stem, file=utt2spk)
            print(path.stem, path.stem, file=spk2utt)
            torchaudio.save(loc, torch.from_numpy(sample['audio']['array'])[None, :], 48000, bits_per_sample=16, encoding='PCM_S')
            print(path.stem, 'sox', str(loc), '-r 16k -t wav -c 1 - |', file=wavscp)

            words = sentence.split()
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


if __name__ == '__main__':
    try:
        auth_token = os.environ['HF_AUTH_TOKEN']
    except KeyError as e:
        raise Exception("""
Request access to the dataset at https://huggingface.co/datasets/mozilla-foundation/common_voice_9_0
and export a token from https://huggingface.co/settings/tokens as HF_AUTH_TOKEN
""") from e

    import argparse
    parser = argparse.ArgumentParser(__file__, description='prepare kaldi data directory with common voice data')
    parser.add_argument('--lexicon', action='store_true', help='generate lexicon for every word using ukro-g2p')
    parser.add_argument('--lang', default='uk', help='language code')
    parser.add_argument('--root', type=Path, default=Path('data/cv'), help='where to put test or train datadirs')
    parser.add_argument('--split', type=str, default='train', help='split to generate (train, validation, test, etc)')

    args = parser.parse_args()

    datadir = args.root / args.split

    uk = datasets.load_dataset('mozilla-foundation/common_voice_9_0', args.lang, split=args.split, use_auth_token=auth_token)
    #
    # or just use an older dataset version (untested):
    #
    #uk = datasets.load_dataset('common_voice', args.lang, split='train+validation')
    #

    if args.lexicon:
        from uk.g2p import g2p
    else:
        g2p = None

    prepare(uk, datadir, g2p=g2p)
