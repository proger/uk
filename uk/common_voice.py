from contextlib import ExitStack
import os
from pathlib import Path
import re

import datasets
from tqdm import tqdm

from uk.g2p import g2p

def keep_useful_characters(s):
    s = s.lower()
    s = s.replace('’', "'")
    s = re.sub(r'[^йцукенгшщзхїфивапролджєґячсміiтьбюЙЦУКЕНГШЩЗХЇФИВАПРОЛДЖЄҐЯЧСМІТЬБЮ\' -]', '', s)
    s = re.sub(r'[ -]+', ' ', s)
    s = re.sub(r'\s+', ' ', s) # unicode whitespace
    s = s.replace('i','і')
    s = s.strip()
    return s


try:
    auth_token = os.environ.get('HF_AUTH_TOKEN')
except KeyError as e:
    raise Exception("""\
Share your contacts at https://huggingface.co/datasets/mozilla-foundation/common_voice_8_0
and export a token from https://huggingface.co/settings/tokens as HF_AUTH_TOKEN
""") from e
else:
    uk = datasets.load_dataset('mozilla-foundation/common_voice_8_0', 'uk', split='train+validation', use_auth_token=auth_token)
finally:
    pass
    #
    # or just use an older dataset version:
    #
    #uk = datasets.load_dataset('common_voice', 'uk', split='train+validation')
    #


def prepare(datadir=Path('data/cv_train')):
    datadir.mkdir(exist_ok=True)

    with ExitStack() as stack:
        text = stack.enter_context(open(datadir / 'text', 'w'))
        utt2spk = stack.enter_context(open(datadir / 'utt2spk', 'w'))
        spk2utt = stack.enter_context(open(datadir / 'spk2utt', 'w'))
        wavscp = stack.enter_context(open(datadir / 'wav.scp', 'w'))
        lexicon_txt = stack.enter_context(open(datadir / 'lexicon.txt', 'w'))

        lexicon = {}

        for sample in tqdm(sorted(uk, key=lambda sample: Path(sample['path']).stem)):
            path = Path(sample['path'])
            sentence = keep_useful_characters(sample['sentence'])
            print(path.stem, sentence, file=text)
            print(path.stem, path.stem, file=utt2spk)
            print(path.stem, path.stem, file=spk2utt)
            print(path.stem, 'sox', str(path), '-r 16k -t wav -c 1 - |', file=wavscp)

            words = sentence.split()
            for word in words:
                if not word in lexicon:
                    lexicon[word] = ' '.join(g2p(word))

        for word in sorted(lexicon):
            print(word, lexicon[word], file=lexicon_txt)


if __name__ == '__main__':
    prepare()
