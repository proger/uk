from pathlib import Path

from uk.subprocess import check_output
from uk.align_utterances import read_symtab


def trim_phone(ali_dir: Path, int_phone, frame_shift=0.01):
    "make segments without leading and trailing phones from utterances in ali_dir"

    ali_to_phones = check_output(['ali-to-phones', '--write-lengths',
                            ali_dir / 'final.alimdl',
                            f'ark:gunzip -c {ali_dir}/ali.*.gz |',
                            f'ark,t:-'])
    for line in ali_to_phones.decode().splitlines():
        # 01-01476000-01479029-1 106 3 ; 283 3 ; 182 6 ; 88 5 ; 296 7
        utt_id, seq = line.split(maxsplit=1)
        int_phones, durations = zip(*[s.split() for s in seq.split(' ; ')])
        end = sum(map(int, durations))
        leading, trailing = 0, end
        state = 'leading'
        for p, d in zip(int_phones, durations):
            if state == 'leading':
                if p == int_phone:
                    leading += int(d)
                else:
                    state = 'inside'
            elif state == 'inside':
                if p == int_phone:
                    trailing -= int(d)
                else:
                    trailing = end
        s, e = leading*frame_shift, trailing*frame_shift
        yield f'{utt_id}-{int(s*100):07d}-{int(e*100):07d}', utt_id, round(s, 2), round(e, 2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""\
    Print segments derived by trimming silence

    python3 -m uk.trim_silence exp/corpus/ali
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--lang-dir', type=Path, default='exp/lang')
    parser.add_argument('ali_dir', type=Path, help='data directory')

    args = parser.parse_args()

    symtab = read_symtab(args.lang_dir)
    rsymtab = {v: k for k, v in symtab.items()}
    for seg in trim_phone(args.ali_dir, rsymtab["sil"]):
        print(*seg)
