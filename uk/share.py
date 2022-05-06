from pathlib import Path

from kaldiio import ReadHelper
import torchaudio
import wandb


def datadir_to_artifact(datadir: Path):
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

    phones = {}
    phone_durations = {}
    if (datadir / 'phones').exists():
        with open(datadir / 'phones') as f:
            for line in f:
                utt_id, phn = line.strip().split(maxsplit=1)
                phones[utt_id] = phn

        with open(datadir / 'phone-durations') as f:
            for line in f:
                utt_id, phn = line.strip().split(maxsplit=1)
                phone_durations[utt_id] = phn

    columns = ["utt_id", "audio", "text"]
    if phones:
        columns.extend(["phones", "phone-durations"])
    table = wandb.Table(columns=columns)

    with ReadHelper(f'scp:{datadir}/wav.scp', segments=segments if segments.exists() else None) as reader:
        for utt_id, (sample_rate, array) in reader:
            audio = wandb.Audio(array, sample_rate=sample_rate)
            row = [utt_id, audio, text[utt_id]]
            if phones:
                row.extend([phones.get(utt_id, ''), phone_durations.get(utt_id, '')])
            table.add_data(*row)

    artifact = wandb.Artifact(datadir.stem, type="dataset")
    artifact.add(table, "index")
    return artifact


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('upload kaldi data to wandb with an index Table', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('datadir', type=Path, help='kaldi data directory')
    args = parser.parse_args()

    run = wandb.init(job_type="dataset")
    run.log_artifact(datadir_to_artifact(args.datadir))
