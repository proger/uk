from pathlib import Path

from kaldiio import ReadHelper
import torchaudio
import wandb


def datadir_to_artifact(datadir: Path):
    columns = ["utt_id", "audio", "duration", "speaker", "text"]
    table = wandb.Table(columns=columns)


    text = {}
    with open(datadir / 'text') as f:
        for line in f:
            utt_id, transcript = line.strip().split(maxsplit=1)
            text[utt_id] = transcript

    speakers = {}
    with open(datadir / 'utt2spk') as f:
        for line in f:
            utt_id, spk = line.strip().split(maxsplit=1)
            speakers[utt_id] = spk

    segments = datadir / 'segments'

    with ReadHelper(f'scp:{datadir}/wav.scp', segments=segments if segments.exists() else None) as reader:
        for utt_id, (sample_rate, array) in reader:
          audio = wandb.Audio(array, sample_rate=sample_rate)
          table.add_data(utt_id, audio, array.shape[-1]/sample_rate, speakers[utt_id], text[utt_id])

    artifact = wandb.Artifact(datadir.stem, type="dataset")
    artifact.add(table, "index")
    return artifact


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('upload kaldi data to wandb with an index Table')
    parser.add_argument('datadir', type=Path, help='kaldi data directory')
    args = parser.parse_args()

    run = wandb.init(job_type="dataset")
    run.log_artifact(datadir_to_artifact(args.datadir))
