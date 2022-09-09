"""
Take a transcript, create kaldi dict using g2p, prepare lang, run steps/cleanup/segment_long_utterances.sh
"""

import argparse
import os
from pathlib import Path

from loguru import logger

from uk.dynamic import import_function
from uk.subprocess import sh
from uk.prepare_lang import extend_dict
from uk.align_utterances import align_utterances, export_alignments, export_as_textgrid, read_symtab


parser = argparse.ArgumentParser(description="""\
python3 -m uk.segment_long_utterances -w exp/corpus -o data/semesyuk_farshrutka_prologue data/local/semesyuk_farshrutka/01_prologue.txt semesyuk-to-text/audio/raw/semesyuk_farshrutka/01_prologue.mp3
""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-w', '--work-dir', type=Path, required=True)
parser.add_argument('-d', '--dict-dir', type=Path, default='exp/dict')
parser.add_argument('-u', '--unk-fst', type=Path, default='exp/dict/unk/unk_fst.txt')
parser.add_argument('-m', '--model-dir', type=Path, default='exp/tri3b')
parser.add_argument('-o', '--output-dir', type=Path, required=True, help='output kaldi data directory with short utterances')
parser.add_argument('--g2p_batch', type=import_function, default='uk.g2p:g2p_batch', help='batched g2p implementation')
parser.add_argument('--nj', default=os.cpu_count() or 1, help='number of parallel jobs')
parser.add_argument('--stage', default=-3, type=int, help='script stage')
parser.add_argument('corpus_txt', type=Path, help='must be tokenized (see README)')
parser.add_argument('mp3', type=Path, help='mp3 to align')

args = parser.parse_args()
stage = args.stage

dict_dir = args.work_dir / 'dict'
dict_dir.mkdir(exist_ok=True, parents=True)

datadir = args.work_dir / 'data'
datadir.mkdir(exist_ok=True, parents=True)

langdir = args.work_dir / 'lang'


if stage <= -3:
    words = args.corpus_txt.read_text().split()
    extend_dict(words, dict_dir, args.dict_dir, g2p_batch=args.g2p_batch)

    if args.unk_fst.is_file() and args.unk_fst.exists():
        sh('utils/prepare_lang.sh', '--unk-fst', args.unk_fst, dict_dir, "<unk>", args.work_dir / 'lang_tmp', langdir)
    else:
        logger.warning('ignored unk-fst')
        sh('utils/prepare_lang.sh', dict_dir, "[unk]", args.work_dir / 'lang_tmp', langdir)

if stage <= -2:
    with open(datadir / 'text', 'w') as f:
        print(args.mp3.stem, ' '.join(words), file=f)

    with open(datadir / 'wav.scp', 'w') as f:
        print(args.mp3.stem, 'ffmpeg -nostdin -i', args.mp3.absolute(), '-ar 16000 -ac 1 -acodec pcm_s16le -f wav - |', file=f)

    with open(datadir / 'utt2spk', 'w') as f:
        print(args.mp3.stem, args.mp3.stem, file=f)

    with open(datadir / 'spk2utt', 'w') as f:
        print(args.mp3.stem, args.mp3.stem, file=f)

if stage <= -1:
    sh('steps/make_mfcc.sh', datadir, mfcc_config='conf/mfcc.conf', nj=1)
    sh('steps/compute_cmvn_stats.sh', datadir)

if stage <= 14:
    sh('steps/cleanup/segment_long_utterances.sh',
        args.model_dir,
        langdir,
        datadir,
        args.work_dir / 'resegmented',
        args.work_dir / 'seg_work',

        max_segment_duration=30,
        overlap_duration=10,
        nj=args.nj,
        stage=args.stage,

        min_split_point_duration=0.3,
        # splitting
        max_segment_length_for_splitting=15, # Try to split long segments into segments that are smaller that this size. See possibly_split_long_segments() in Segment class
        hard_max_segment_length=20, # Split all segments that are longer than this uniformly into segments of size
        min_silence_length_to_split_at=0.1, # Only considers silences that are at least this long as potential split points"
        max_bad_proportion=1.0,
        max_wer=100,
        # extra
        segmentation_extra_opts='--min-new-segment-length 0.01 --min-segment-length 0.01 --max-internal-silence-length 1.0 --max-internal-non-scored-length 5.0 --max-edge-non-scored-length 1.0 --max-edge-silence-length 1.0 --max-tainted-length 1.0'
        )

if stage <= 16:
    # redefine wav.scp to use fullband wav instead
    with open(args.work_dir / 'resegmented' / 'wav.scp', 'w') as f:
        print(args.mp3.stem, 'ffmpeg -nostdin -i', args.mp3.absolute(), '-ac 1 -acodec pcm_s16le -f wav - |', file=f)

    # extract each segments as its own wav into the output
    (args.output_dir / 'wav').mkdir(exist_ok=True)
    with open(args.output_dir / 'wav.scp', 'w') as out:
        with open(args.work_dir / 'resegmented' / 'segments') as f:
            for line in f:
                segment_id = line.split()[0]
                print(segment_id, args.output_dir / 'wav' / f'{segment_id}.wav', file=out)

    sh('extract-segments',
       f"scp:{args.work_dir / 'resegmented' / 'wav.scp'}",
       args.work_dir / 'resegmented' / 'segments',
       f"scp:{args.output_dir / 'wav.scp'}")

if stage <= 17:
    align_utterances(args.work_dir / 'resegmented', langdir, args.model_dir, args.work_dir / 'ali', nj=args.nj)

symtab = read_symtab(langdir)

if stage <= 18:
    export_alignments(args.work_dir / 'ali', symtab, args.output_dir)

if stage <= 19:
    export_as_textgrid(args.work_dir / 'ali', symtab, args.output_dir)

if stage <= 20:
    logger.info('upload to wandb using: python3 -m uk.share {}', args.output_dir)
