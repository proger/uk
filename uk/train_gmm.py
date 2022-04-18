import os

from uk.subprocess import sh

nproc = os.cpu_count() or 1

stage = 0

datadir = 'data/cv_train'
langdir = 'data/lang'


if stage <= 0:
    sh('local/prepare_dict.sh')

if stage <= 1:
    sh('utils/prepare_lang.sh', 'data/local/dict', '<unk>', 'data/local/lang', langdir)

if stage <= 2:
    sh('steps/make_mfcc.sh', datadir, 'exp/log/mfcc/cv_train', 'exp/mfcc/cv_train', mfcc_config='conf/mfcc.conf', nj=nproc)

if stage <= 3:
    sh('steps/compute_cmvn_stats.sh', datadir, 'exp/log/mfcc/cv_train', 'exp/mfcc/cv_train')

if stage <= 4:
    sh('steps/train_mono.sh', datadir, langdir, 'exp/mono', boost_silence=1.25, nj=nproc)

if stage <= 5:
    pass

if stage <= 6:
    sh('steps/align_si.sh', datadir, langdir, 'exp/mono', 'exp/mono_ali', boost_silence=1.25, nj=nproc)

if stage <= 7:
    sh('steps/train_deltas.sh', 2000, 10000, datadir, langdir, 'exp/mono_ali', 'exp/tri1', boost_silence=1.25)

if stage <= 8:
    # decode?
    pass

if stage <= 9:
    sh('steps/align_si.sh', datadir, langdir, 'exp/tri1', 'exp/tri1_ali', nj=nproc)

if stage <= 10:
    sh('steps/train_lda_mllt.sh', 2500, 15000, datadir, langdir, 'exp/tri1_ali', 'exp/tri2b', splice_opts="--left-context=3 --right-context=3")

if stage <= 11:
    # decode?
    pass

if stage <= 12:
    sh('steps/align_si.sh', datadir, langdir, 'exp/tri2b', 'exp/tri2b_ali', use_graphs='true', nj=nproc)

if stage <= 13:
    sh('steps/train_sat.sh', 2500, 15000, datadir, langdir, 'exp/tri2b_ali', 'exp/tri3b')

if stage <= 14:
    # decode?
    pass