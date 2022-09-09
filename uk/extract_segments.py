from pathlib import Path

from uk.subprocess import sh


def extract_segments(output_data_dir: Path, source_wav_scp: Path, segments_file: Path):
    "extract each segment as its own wav into the output"

    (output_data_dir / 'wav').mkdir(exist_ok=True, parents=True)
    with open(output_data_dir / 'wav.scp', 'w') as out:
        with open(segments_file) as f:
            for line in f:
                segment_id = line.split()[0]
                print(segment_id, output_data_dir / 'wav' / f'{segment_id}.wav', file=out)

    sh('extract-segments',
       f"scp:{source_wav_scp}",
       segments_file,
       f"scp:{output_data_dir / 'wav.scp'}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""\
    Extract each segment as its own wav into the new data directory.

    python3 -m uk.extract_segments exp/corpus/ali
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output-data-dir', type=Path)
    parser.add_argument('-i', '--source-wav-scp', type=Path)
    parser.add_argument('segments', type=Path, help='segments_file')

    args = parser.parse_args()

    extract_segments(args.output_data_dir, args.source_wav_scp, args.segments)