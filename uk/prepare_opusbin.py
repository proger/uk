"""
Explore nested directories of media files and produce a binary file with
16kbps raw Opus CELT packets (60ms frame duration).

python3 -m uk.prepare_opusbin INPUT_DIR OUTPUT_FILE
"""

import argparse
import sys
from pathlib import Path

from uk.subprocess import check_output


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--ext', default="webm", type=str,
        help='media file extension'
    )
    parser.add_argument(
        'input_dir', type=Path,
        help='root directory to search for media files'
    )
    parser.add_argument(
        'output_file', type=Path,
        help='binary output file for raw Opus packets'
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file

    paths = sorted(input_dir.rglob(f'*.{args.ext}'))
    if not paths:
        sys.stderr.write(f'No media files found in {input_dir}\n')
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total = len(paths)
    n_done = 0
    with open(output_file, 'wb') as fout:
        for p in paths:
            try:
                opus_bytes = check_output([
                    'ffmpeg', '-hide_banner', '-loglevel', 'error',
                    '-i', str(p),
                    '-map', '0:a',
                    '-c:a', 'libopus',
                    '-b:a', '16k',
                    '-vbr', 'off',
                    '-application', 'voip',
                    '-frame_duration', '60',
                    '-f', 'data',
                    '-',
                ])
            except Exception as e:
                sys.stderr.write(f'Error encoding {p}: {e}\n')
                sys.exit(1)

            fout.write(opus_bytes)
            n_done += 1
            if n_done % 10 == 0 or n_done == total:
                print(f'Encoded {n_done}/{total} files', end='\r', flush=True)

    print(f'\nDone: encoded {n_done} files to {output_file}')


if __name__ == '__main__':
    main()
