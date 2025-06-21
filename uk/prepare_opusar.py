"""
prepare_mkopusutf.py: stage-based pipeline for building Opus+UTF dataset.

Stages:
 0) Encode audio to raw Opus+transcript messages (mkopusutf)
 1) Print ASCII histogram of transcript lengths
 2) Print ASCII histogram of message (byte) lengths
 3) Filter messages by maximum length
 4) Pack filtered messages into fixed-length sequences with optimal bin-packing and pad zeros
 5) Print ASCII histogram of packed sequence lengths

This script supports random speed augmentation in stage 0 (via --epochs) with speed
factors uniformly sampled between 0.95 and 1.1, as well as dynamic range compression,
volume perturbations, and slight pitch shifts. Effects are applied via a mixture-of-experts
approach: 1–2 augmentations are chosen at random per utterance. You can disable all
audio augmentations in stage 0 by passing --disable-augmentation.
"""

import argparse
import sys
from pathlib import Path

from uk.subprocess import check_output
import struct
import random


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--stage', type=int, default=0,
                   help='pipeline stage to start from')
    p.add_argument('--data', required=True,
                   help='input dir containing wav.scp and text files')
    p.add_argument('--exp', required=True,
                   help='experiment directory for outputs (messages.bin, lengths.bin, etc)')
    p.add_argument('--hist-bins', type=int, default=10,
                   help='number of bins for histograms')
    p.add_argument('--filter-max-bytes', type=int, default=16384,
                   help='max message bytes to keep (filter stage)')
    p.add_argument('--seq-length', type=int, default=16384,
                   help='target sequence length for packing messages (bytes)')
    p.add_argument('--repeat-factor', type=int, default=1,
                   help='repeat training data this many times with different packings')
    p.add_argument('--disable-packing', action='store_true',
                   help='disable bin-packing in stage 4, only pad messages individually to seq_length')
    p.add_argument('--epochs', type=int, default=1,
                   help='number of times to repeat data augmentation with randomized speed perturbations in stage 0')
    p.add_argument('--disable-augmentation', action='store_true',
                   help='disable all audio augmentations in stage 0')
    return p.parse_args()


def load_transcripts(path):
    transcripts = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            utt, txt = line.split(' ', 1)
            transcripts[utt] = txt
    return transcripts


def stage0_encode(wav_scp, text_path, out_data, out_lens, epochs, disable_augmentation=False):
    transcripts = load_transcripts(text_path)
    n_done = 0
    with open(out_data, 'wb') as fout_data, open(out_lens, 'wb') as fout_lens:
        for ep in range(epochs):
            with open(wav_scp, 'r', encoding='utf-8') as fscp:
                for line in fscp:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ', 1)
                    if len(parts) != 2:
                        continue
                    utt, cmd = parts
                    if utt not in transcripts:
                        continue
                    text = transcripts[utt]

                    sox_cmd = cmd.strip()
                    if disable_augmentation:
                        ffmpeg_cmd = (
                            'ffmpeg -hide_banner -loglevel error '
                            '-f wav -i - '
                            '-map 0:a '
                            '-c:a libopus -b:a 16k -vbr off '
                            '-application voip -frame_duration 60 '
                            '-f data -'
                        )
                    else:
                        # Sample augmentation parameters
                        speed = random.uniform(0.95, 1.1)
                        volume = random.uniform(0.8, 1.2)
                        pitch = random.uniform(0.9, 1.1)
                        comp = 'acompressor=threshold=-20dB:ratio=3:attack=5:release=50'
                        # Mixture-of-experts: pick 1-2 augmentations at random
                        opts = {
                            'pitch': f'asetrate=sr*{pitch:.3f},aresample=sr',
                            'comp': comp,
                            'volume': f'volume={volume:.3f}',
                            'speed': f'atempo={speed:.3f}'
                        }
                        n_fx = random.choice([1, 2])
                        selected = random.sample(list(opts.values()), n_fx)
                        filter_chain = ','.join(selected)
                        ffmpeg_cmd = (
                            'ffmpeg -hide_banner -loglevel error '
                            '-f wav -i - '
                            f'-filter:a "{filter_chain}" '
                            '-map 0:a '
                            '-c:a libopus -b:a 16k -vbr off '
                            '-application voip -frame_duration 60 '
                            '-f data -'
                        )
                    full_cmd = f"{sox_cmd} {ffmpeg_cmd}"
                    try:
                        opus_bytes = check_output(
                            full_cmd, shell=True, executable='/bin/bash'
                        )
                    except Exception as e:
                        sys.stderr.write(f"Error encoding {utt}: {e}\n")
                        sys.exit(1)

                    msg = b'\x01' + opus_bytes + (b'\x07' * 16) + text.encode('utf-8') + b'\x04'
                    fout_data.write(msg)
                    fout_lens.write(struct.pack('>I', len(msg)))

                    n_done += 1
                    if n_done % 100 == 0:
                        print(f"Encoded {n_done} utts", end='\r', flush=True)
    print(f"Stage 0 done: encoded {n_done} utterances (epochs={epochs})\n")


def stage1_hist_texts(text_path, bins):
    transcripts = load_transcripts(text_path)
    lengths = [len(t) for t in transcripts.values()]
    if not lengths:
        print('No transcripts to histogram')
        return
    lo, hi = min(lengths), max(lengths)
    span = hi - lo + 1
    width = span / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for L in lengths:
        idx = int((L - lo) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    maxc = max(counts)
    bar_max = 50
    print(f"Transcript length histogram ({len(lengths)} items):")
    for i in range(bins):
        start = int(edges[i])
        end = int(edges[i+1] - 1) if i < bins - 1 else hi
        bar_len = int(counts[i] / maxc * bar_max) if maxc else 0
        print(f"{start:5d}-{end:5d} | {'*' * bar_len} {counts[i]}")


def stage2_hist_msgs(lens_path, bins):
    lengths = []
    with open(lens_path, 'rb') as f:
        while True:
            data = f.read(4)
            if not data:
                break
            if len(data) != 4:
                break
            lengths.append(struct.unpack('>I', data)[0])
    if not lengths:
        print('No message lengths to histogram')
        return
    lo, hi = min(lengths), max(lengths)
    span = hi - lo + 1
    width = span / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for L in lengths:
        idx = int((L - lo) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    maxc = max(counts)
    bar_max = 50
    print(f"Message byte-length histogram ({len(lengths)} items):")
    for i in range(bins):
        start = int(edges[i])
        end = int(edges[i+1] - 1) if i < bins - 1 else hi
        bar_len = int(counts[i] / maxc * bar_max) if maxc else 0
        print(f"{start:7d}-{end:7d} | {'*' * bar_len} {counts[i]}")


def stage3_filter(in_data, in_lens, max_bytes, out_data, out_lens):
    kept = total = 0
    with open(in_data, 'rb') as fdata, open(in_lens, 'rb') as flens, \
         open(out_data, 'wb') as od, open(out_lens, 'wb') as ol:
        while True:
            lb = flens.read(4)
            if not lb:
                break
            if len(lb) != 4:
                sys.stderr.write('Warning: partial length, stopping\n')
                break
            L = struct.unpack('>I', lb)[0]
            msg = fdata.read(L)
            if len(msg) != L:
                sys.stderr.write('Warning: unexpected EOF in data\n')
                break
            total += 1
            if L <= max_bytes:
                od.write(msg)
                ol.write(lb)
                kept += 1
    sys.stderr.write(f"Stage 3 done: kept {kept}/{total} messages <= {max_bytes} bytes\n")

def stage4_pack(in_data, in_lens, seq_length, repeat_factor, disable_packing, out_data, out_lens):
    """
    Pack messages into fixed-size sequences of length seq_length (bytes) using a best-fit bin-packing heuristic.
    If disable_packing is True, skip bin-packing and only pad each message individually.
    Repeat the packing or padding process repeat_factor times.
    Pads each sequence to exactly seq_length with zeros.
    Writes packed data to out_data and original sequence lengths (before padding) to out_lens.
    """
    lengths = []
    messages = []
    with open(in_lens, 'rb') as flens, open(in_data, 'rb') as fdata:
        while True:
            lb = flens.read(4)
            if not lb:
                break
            if len(lb) != 4:
                sys.stderr.write('Warning: partial length in pack lens, stopping\n')
                break
            L = struct.unpack('>I', lb)[0]
            msg = fdata.read(L)
            if len(msg) != L:
                sys.stderr.write('Warning: unexpected EOF in data for packing\n')
                break
            lengths.append(L)
            messages.append(msg)
    total_msgs = len(messages)
    if total_msgs == 0:
        sys.stderr.write('Stage 4: no messages to pack\n')
        return
    if disable_packing:
        total_seqs = 0
        with open(out_data, 'wb') as od, open(out_lens, 'wb') as ol:
            for rep in range(repeat_factor):
                sys.stderr.write(f"Padding repetition {rep+1}/{repeat_factor}\n")
                for i, msg in enumerate(messages):
                    L = lengths[i]
                    if L > seq_length:
                        sys.stderr.write(f"Error: message length {L} > {seq_length}\n")
                        sys.exit(1)
                    seq = msg
                    if L < seq_length:
                        seq += b'\x00' * (seq_length - L)
                    od.write(seq)
                    ol.write(struct.pack('>I', L))
                    total_seqs += 1
        sys.stderr.write(f"Stage 4 done (padding-only): padded into {total_seqs} sequences "
                         f"(seq_length={seq_length}, repeat_factor={repeat_factor})\n")
        return

    total_seqs = 0
    with open(out_data, 'wb') as od, open(out_lens, 'wb') as ol:
        for rep in range(repeat_factor):
            sys.stderr.write(f"Packing repetition {rep+1}/{repeat_factor}\n")
            idxs = list(range(total_msgs))
            random.shuffle(idxs)
            bins_remain = []
            bins_msgs = []
            for idx in idxs:
                L = lengths[idx]
                best_i = None
                best_rem = None
                for i, rem in enumerate(bins_remain):
                    if rem >= L:
                        rem_after = rem - L
                        if best_i is None or rem_after < best_rem:
                            best_i = i
                            best_rem = rem_after
                if best_i is None:
                    bins_msgs.append([idx])
                    bins_remain.append(seq_length - L)
                else:
                    bins_msgs[best_i].append(idx)
                    bins_remain[best_i] = best_rem
            for b_msgs, rem in zip(bins_msgs, bins_remain):
                seq = b''.join(messages[i] for i in b_msgs)
                seq_len = len(seq)
                if seq_len > seq_length:
                    sys.stderr.write(f"Error: packed seq length {seq_len} > {seq_length}\n")
                    sys.exit(1)
                if seq_len < seq_length:
                    seq += b'\x00' * (seq_length - seq_len)
                od.write(seq)
                ol.write(struct.pack('>I', seq_len))
                total_seqs += 1
    sys.stderr.write(f"Stage 4 done: packed into {total_seqs} sequences "
                     f"(seq_length={seq_length}, repeat_factor={repeat_factor})\n")

def stage5_hist_packed(lens_path, bins):
    """
    Print an ASCII histogram of packed sequence lengths from lens_path.
    """
    lengths = []
    with open(lens_path, 'rb') as f:
        while True:
            data = f.read(4)
            if not data:
                break
            if len(data) != 4:
                break
            lengths.append(struct.unpack('>I', data)[0])
    if not lengths:
        print('No packed sequence lengths to histogram')
        return
    lo, hi = min(lengths), max(lengths)
    span = hi - lo + 1
    width = span / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for L in lengths:
        idx = int((L - lo) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    maxc = max(counts)
    bar_max = 50
    print(f"Packed sequence length histogram ({len(lengths)} items):")
    for i in range(bins):
        start = int(edges[i])
        end = int(edges[i+1] - 1) if i < bins - 1 else hi
        bar_len = int(counts[i] / maxc * bar_max) if maxc else 0
        print(f"{start:7d}-{end:7d} | {'*' * bar_len} {counts[i]}")


def main():
    args = parse_args()
    st = args.stage

    exp = Path(args.exp)
    exp.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    wav_scp = data / 'wav.scp'
    text_path = data / 'text'
    out_data = exp / 'messages.bin'
    out_lens = exp / 'lengths.bin'
    filtered_data = exp / 'filtered_messages.bin'
    filtered_lens = exp / 'filtered_lengths.bin'
    packed_data = exp / 'packed_messages.bin'
    packed_lens = exp / 'packed_lengths.bin'

    if st <= 0:
        stage0_encode(
            wav_scp, text_path, out_data, out_lens,
            args.epochs, args.disable_augmentation
        )
    if st <= 1:
        stage1_hist_texts(text_path, args.hist_bins)
    if st <= 2:
        stage2_hist_msgs(out_lens, args.hist_bins)
    if st <= 3 and args.filter_max_bytes is not None:
        stage3_filter(
            out_data, out_lens,
            args.filter_max_bytes,
            filtered_data, filtered_lens
        )

    if st <= 4:
        stage4_pack(
            filtered_data, filtered_lens,
            args.seq_length, args.repeat_factor, args.disable_packing,
            packed_data, packed_lens
        )

    if st <= 5:
        stage5_hist_packed(packed_lens, args.hist_bins)


if __name__ == '__main__':
    main()
