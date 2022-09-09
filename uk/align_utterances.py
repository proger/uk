from loguru import logger

from uk.subprocess import sh, check_output
from uk.textgrid import ctm_to_textgrid


def read_symtab(langdir):
    "read phone symbol table"
    symtab = {}
    with open(langdir / 'phones.txt') as f:
        for line in f:
            phone_sym, phone_int = line.split()
            symtab[phone_int] = phone_sym
    return symtab


def align_utterances(input_dir, langdir, model_dir, ali_dir, nj=1):
    sh('steps/compute_cmvn_stats.sh', input_dir)
    sh('steps/align_fmllr.sh', input_dir, langdir, model_dir, ali_dir, nj=nj)


def export_alignments(ali_dir, symtab, output_dir):
    phones = {}
    phone_durations = {}

    ali_to_phones = check_output(['ali-to-phones', '--write-lengths',
                                ali_dir / 'final.alimdl',
                                f'ark:gunzip -c {ali_dir}/ali.*.gz |',
                                f'ark,t:-'])
    for line in ali_to_phones.decode().splitlines():
        # 01-01476000-01479029-1 106 3 ; 283 3 ; 182 6 ; 88 5 ; 296 7
        utt_id, seq = line.split(maxsplit=1)
        phones1, durations = zip(*[s.split() for s in seq.split(' ; ')])
        phones[utt_id] = ' '.join(symtab[phone_int] for phone_int in phones1)
        phone_durations[utt_id] = ' '.join(durations)

    with open(output_dir / 'phones', 'w') as f:
        for utt_id in sorted(phones):
            print(utt_id, phones[utt_id], file=f)

    with open(output_dir / 'phone-durations', 'w') as f:
        for utt_id in sorted(phone_durations):
            print(utt_id, phone_durations[utt_id], file=f)


def export_as_textgrid(ali_dir, symtab, output_dir):
    ctm_output = output_dir / 'ctm'
    check_output(['ali-to-phones', '--ctm-output',
                ali_dir / 'final.alimdl',
                f'ark:gunzip -c {ali_dir}/ali.*.gz |',
                ctm_output])
    logger.info('wrote ctm to {}', ctm_output)
    textgrid_output = output_dir / 'textgrid'
    textgrid_output.mkdir(exist_ok=True)
    with open(ctm_output) as f:
        ctm_to_textgrid(f, textgrid_output, symtab)
    logger.info('wrote TextGrid files to {}', textgrid_output)


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="""\
    Align utterances for downstream applications.

    python3 -m uk.align_utterances -a exp/corpus/ali -l exp/corpus/lang -m exp/tri3b -o data/semesyuk_farshrutka_prologue exp/corpus/resegmented
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--output-ali-dir', type=Path, required=True)
    parser.add_argument('-l', '--lang-dir', type=Path, default='exp/lang')
    parser.add_argument('-m', '--model-dir', type=Path, default='exp/tri3b')
    parser.add_argument('data_dir', type=Path, help='data directory')

    args = parser.parse_args()

    align_utterances(args.data_dir, args.lang_dir, args.model_dir, args.output_ali_dir)
    symtab = read_symtab(args.lang_dir)
    export_alignments(args.output_ali_dir, symtab, args.data_dir)
    export_as_textgrid(args.output_ali_dir, symtab, args.data_dir)

    logger.info('upload alignments to wandb using: python3 -m uk.share {}', args.data_dir)
