"""
Take a transcript, create kaldi dict using g2p, prepare lang, run steps/cleanup/segment_long_utterances.sh
"""

import argparse
from pathlib import Path
import shutil
from typing import List

from uk.g2p import g2p
from uk.subprocess import sh

parser = argparse.ArgumentParser(description="""\
python3 -m uk.segment_long_utterances -w exp/corpus -o data/semesyuk_farshrutka_prologue data/local/semesyuk_farshrutka/01_prologue.txt semesyuk-to-text/audio/raw/semesyuk_farshrutka/01_prologue.mp3
""")
parser.add_argument('-w', '--work-dir', type=Path, required=True)
parser.add_argument('-d', '--dict-dir', type=Path, default='data/local/dict')
parser.add_argument('-u', '--unk-fst', type=Path, default='exp/make_unk/unk_fst.txt')
parser.add_argument('-m', '--model-dir', type=Path, default='exp/tri3b')
parser.add_argument('-o', '--output-dir', type=Path, required=True, help='output kaldi data directory with short utterances')
parser.add_argument('corpus_txt', type=Path, help='must be tokenized (see README)')
parser.add_argument('mp3', type=Path, help='mp3 to align')

args = parser.parse_args()

dict_dir = args.work_dir / 'dict'
dict_dir.mkdir(exist_ok=True, parents=True)


def extend_dict(words: List[str], dict_dir: Path, source_dict_dir: Path = args.dict_dir):
    shutil.copy(source_dict_dir / 'extra_questions.txt', dict_dir / 'extra_questions.txt')
    shutil.copy(source_dict_dir / 'optional_silence.txt', dict_dir / 'optional_silence.txt')
    shutil.copy(source_dict_dir / 'silence_phones.txt', dict_dir / 'silence_phones.txt')
    shutil.copy(source_dict_dir / 'nonsilence_phones.txt', dict_dir / 'nonsilence_phones.txt')

    with open(dict_dir / 'lexicon.txt', 'w') as f:
        oov = {}
        for word in words:
            oov[word] = ' '.join(g2p(word))

        with open(source_dict_dir / 'lexicon.txt') as lexicon:
            for line in lexicon:
                word, prons = line.split(maxsplit=1)
                print(word, prons.strip(), file=f)
                if word in oov:
                    del oov[word]

        for word in oov:
            print(word, oov[word].strip(), file=f)


words = args.corpus_txt.read_text().split()
extend_dict(words, dict_dir)

langdir = args.work_dir / 'lang'
sh('utils/prepare_lang.sh', '--unk-fst', args.unk_fst, dict_dir, "<unk>", args.work_dir / 'lang_tmp', langdir)

datadir = args.work_dir / 'data'
datadir.mkdir(exist_ok=True, parents=True)

with open(datadir / 'text', 'w') as f:
    print(args.mp3.stem, ' '.join(words), file=f)

with open(datadir / 'wav.scp', 'w') as f:
    print(args.mp3.stem, 'ffmpeg -nostdin  -i', args.mp3.absolute(), '-ar 16000 -ac 1 -acodec pcm_s16le -f wav - |', file=f)

with open(datadir / 'utt2spk', 'w') as f:
    print(args.mp3.stem, args.mp3.stem, file=f)

with open(datadir / 'spk2utt', 'w') as f:
    print(args.mp3.stem, args.mp3.stem, file=f)

sh('steps/make_mfcc.sh', datadir, mfcc_config='conf/mfcc.conf', nj=1)
sh('steps/compute_cmvn_stats.sh', datadir)

sh('steps/cleanup/segment_long_utterances.sh',
    args.model_dir,
    langdir,
    datadir,
    args.work_dir / 'resegmented',
    args.work_dir / 'seg_work',
    max_segment_duration=10,
    overlap_duration=5)

sh('utils/data/extract_wav_segments_data_dir.sh',  args.work_dir / 'resegmented', args.output_dir)

output_wav_dir = args.output_dir / 'wav'
output_wav_dir.mkdir(exist_ok=True, parents=True)

with open(args.output_dir / 'wav.scp') as f:
    with open(args.output_dir / 'export.scp', 'w') as w:
        for line in f:
            utt_id, _ = line.strip().split(maxsplit=1)
            print(utt_id, (output_wav_dir / utt_id).with_suffix('.wav'), file=w)

sh('wav-copy', f"scp:{args.output_dir / 'wav.scp'}", f"scp:{args.output_dir / 'export.scp'}")
