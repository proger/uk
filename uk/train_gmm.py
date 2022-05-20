import argparse
import os
from pathlib import Path

from uk.subprocess import sh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""\
    Train GMM models.
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--datadir', type=Path, default='data/cv/train', metavar='data/path/to/train', help='train data directory (see uk.common_voice)')
    parser.add_argument('--dictdir', type=Path, metavar='data/local/dict', default='data/local/dict', help='dictionary directory (see local/prepare_dict.sh)')
    parser.add_argument('--unk', metavar='<unk>', default='<unk>', help='unk word (could be [unk])')
    parser.add_argument('--stage', type=int, metavar='0', default=0)
    parser.add_argument('exp', type=Path, help='experiment root directory')
    args = parser.parse_args()
else:
    raise Exception('this module is a script')


nproc = os.cpu_count() or 1
stage = args.stage
datadir = args.datadir
langdir = args.exp / 'lang'

if stage <= 0:
    sh('local/prepare_dict.sh')

if stage <= 1:
    sh('utils/prepare_lang.sh', args.dictdir, args.unk, args.exp / 'langtmp', langdir)

if stage <= 2:
    sh('steps/make_mfcc.sh', datadir, args.exp / 'log/mfcc/train', args.exp / 'mfcc/train', mfcc_config='conf/mfcc.conf', nj=nproc)

if stage <= 3:
    sh('steps/compute_cmvn_stats.sh', datadir, args.exp / 'log/mfcc/train', args.exp / 'mfcc/train')

if stage <= 4:
    sh('steps/train_mono.sh', datadir, langdir, args.exp / 'mono', boost_silence=1.25, nj=nproc)

if stage <= 5:
    pass

if stage <= 6:
    sh('steps/align_si.sh', datadir, langdir, args.exp / 'mono', args.exp / 'mono_ali', boost_silence=1.25, nj=nproc)

if stage <= 7:
    sh('steps/train_deltas.sh', 2000, 10000, datadir, langdir, args.exp / 'mono_ali', args.exp / 'tri1', boost_silence=1.25)

if stage <= 8:
    # decode?
    pass

if stage <= 9:
    sh('steps/align_si.sh', datadir, langdir, args.exp / 'tri1', args.exp / 'tri1_ali', nj=nproc)

if stage <= 10:
    sh('steps/train_lda_mllt.sh', 2500, 15000, datadir, langdir, args.exp / 'tri1_ali', args.exp / 'tri2b', splice_opts="--left-context=3 --right-context=3")

if stage <= 11:
    # decode?
    pass

if stage <= 12:
    sh('steps/align_si.sh', datadir, langdir, args.exp / 'tri2b', args.exp / 'tri2b_ali', use_graphs='true', nj=nproc)

if stage <= 13:
    sh('steps/train_sat.sh', 2500, 15000, datadir, langdir, args.exp / 'tri2b_ali', args.exp / 'tri3b')

if stage <= 14:
    # decode?
    pass
