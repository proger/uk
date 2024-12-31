#%%
import matplotlib
matplotlib.use("kitcat")
import matplotlib.pyplot as plt
import pydub
from pathlib import Path
import numpy as np
from tqdm.contrib.concurrent import thread_map
import time
import io
import contextlib

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
SPACE = '␣'



def rewrite_text(s):
    return SPACE + SPACE.join(s.split(' ')) + SPACE

def read_table(file_path, skip=0, max_length=10000):
    entries = {}
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i < skip:
                continue
            key, text = line.strip().split(maxsplit=1)
            text = rewrite_text(text)
            if len(text) < max_length:
                entries[key] = text
    return entries

wav_dir = Path('wav')
transcripts = read_table(Path('data/mozilla-foundation/common_voice_10_0/uk/train/text'), max_length=30)
train_keys = list(transcripts.keys())

SAMPLING_RATE = 16000
FPS = 100
WINDOW_SIZE = 320
HOP_SIZE = 160


def preemph(x, c=0.97):
    return np.append(x[0], x[1:] - c * x[:-1])

def hann_window(size=WINDOW_SIZE):
    n = np.arange(size)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (size - 1)))

def dft(size=WINDOW_SIZE, phase=0):
    k = np.arange(size) / (2 * size)
    t = np.arange(size)
    phase *= 2j * np.pi * k[:, None]
    return np.exp(-2j * np.pi * k[:, None] * t + phase)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def warp(f, finv, out_dim, in_dim, a, b):
    src = finv(np.linspace(f(a), f(b), out_dim + 2))
    src = np.floor(in_dim * src / b).astype(int)

    map = np.zeros((out_dim, in_dim))

    for i in range(1, out_dim + 1):
        l, c, r = src[i-1], src[i], src[i+1]
        map[i-1, l:c] = (np.arange(l, c) - l) / (c - l)
        map[i-1, c:r] = (r - np.arange(c, r)) / (r - c)
    
    return map

def cmvn(frames):
    frames = frames - np.mean(frames, axis=0, keepdims=True)
    frames = frames / np.std(frames, axis=0, keepdims=True)
    return frames

for key in train_keys:
    break

def slow_print(text, delay=0.01):
    for c in text.split(' '):
        print(c, flush=True, end=' ')
        time.sleep(delay)

class Beats:
    def __init__(self, time=0.5):
        self.time = time
        self.i = 0
        self.frames = []
        self.buffers = []

    def __enter__(self):
        self.delayed = True
        self.buf = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.buf)
        self.redirect.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.redirect.__exit__(exc_type, exc_val, exc_tb)
        self.buffers.append(self.buf.getvalue())
        pass

    def next_name(self):
        name = f'exp/frame_{self.i}.png'
        self.i += 1
        self.frames.append(name)
        return name

    def replay(self):
        for text, filename in zip(self.buffers, self.frames):
            import os
            start = time.time()
            #print('\033[2J\033[H', end='')
            slow_print(text)
            end = time.time()
            sleep = self.time - (end - start)
            #print(sleep, 'sleep time')
            time.sleep(max(0, sleep))
            os.system(f'kitty icat {filename}')
            
    def plot(self, *xs, title=''):
        fig, ax = plt.subplots(1,1, figsize=(20, 5))
        fig.set_tight_layout(True)
        for x in xs:
            ax.plot(x)
        if title:
            print(title)

        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.savefig(self.next_name())
        plt.close(fig)

    def plots(self, *xs, title='', delayed=False):
        xs = list(xs)
        fig, axes = plt.subplots(1, len(xs), figsize=(20, 5))
        fig.set_tight_layout(True)
        if len(xs) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        if title:
            print(title)
        for ax, x in zip(axes, xs):
            ax.plot(x)
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.savefig(self.next_name())
        plt.close(fig)

    def matshow(self, *xs, title=''):
        xs = list(xs)
        fig, axes = plt.subplots(1, len(xs), figsize=(20, 5))
        fig.set_tight_layout(True)
        if len(xs) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        if title:
            print(title)
        for ax, x in zip(axes, xs):
            ax.matshow(x.T, aspect='auto')
            for spine in ax.spines.values():
                spine.set_visible(False)
        plt.savefig(self.next_name())
        plt.close(fig)

def samples(audio):
    samples = np.array(audio.get_array_of_samples())
    range_max = 2**(audio.sample_width*8-1)
    samples = samples.astype(np.float32) / range_max
    return samples

beat = Beats()

slow_print("Let's animate the process of extracting a 13 Mel-Frequency Cepstral Coefficients (MFCC) spectrogram from an MP3 file.\n")
with beat as b:
    name = str(wav_dir / f'{key}.mp3')
    x = pydub.AudioSegment.from_mp3(name)
    x = x.set_frame_rate(16000)
    print('# Read and resample the input')
    print(f'x = samples(pydub.AudioSegment.from_mp3({repr(name)}).set_frame_rate(16000))')
    b.plot(samples(x))
with beat as b:
    print('# Normalize signal amplitude')
    x = samples(x)
    x = x / np.max(x)
    b.plot(x, title='x = x / np.max(x)')
with beat as b:
    print('# Pre-emphasize')
    x = preemph(x)
    b.plot(x, title='x = x[1:] - 0.97 * x[:-1]')
# with beat:
#     x = x / np.max(x)
#     b.plot(x, title='x = x / np.max(x)')
with beat as b:
    print(f'# Frame the signal using a sliding window of length {WINDOW_SIZE}')
    x = np.lib.stride_tricks.sliding_window_view(x, WINDOW_SIZE)[::HOP_SIZE]
    b.plot(x.T, title='x = np.lib.stride_tricks.sliding_window_view(x, WIN)[::WIN//2]')
with beat as b:
    print(f'# Construct Hann smoothing window')
    window = hann_window()
    b.plot(x.T, window, title='hann_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(WIN) / (size - 1)))')
with beat as b:
    print(f'# Apply Hann window')
    x = window * x
    b.plot(x.T, title='x = hann_window * x')
with beat as b:
    print(f'# Construct Discrete Fourier Transform (DFT) basis')
    print('k = np.arange(WIN) / (2 * WIN)')
    print('t = np.arange(WIN)')
    print('dft_basis = np.exp(-2j * np.pi * k[:, None] * t)')
    dft_basis = dft()
    b.plots(dft_basis.real, dft_basis.imag)
with beat as b:
    print(f'# Collect frequency statistics from each frame')
    x = np.abs(x @ dft_basis.T)
    b.matshow(x, title='x = np.abs(x @ dft_basis.T)')
with beat as b:
    print(f'# Take the power of the spectrum magnitude')
    x = x**2
    b.matshow(x, title='x = x**2')
with beat as b:
    print(f'# Construct the perceptual mel scale')
    mel_basis = warp(hz_to_mel, mel_to_hz, 40, 320, a=50, b=16000)
    print('mel_basis = warp(lambda hz: 2595 * np.log10(1 + hz / 700), lambda mel: 700 * (10**(mel / 2595) - 1), 40, 320, a=50, b=16000)')
    b.matshow(mel_basis.T)
with beat as b:
    print('# Warp power spectrum to the mel scale')
    print('x = x @ mel_basis.T')
    x = x @ mel_basis.T
    b.matshow(x)
with beat as b:
    print('# Add noise floor and move to the log space')
    print('x = np.log10(x + 1e-2)')
    x = np.log10(x + 1e-2)
    b.matshow(x)
with beat as b:
    print('# Construct the Discrete Cosine Transform basis')
    print('dct_basis = np.exp(-2j * np.pi * k[:, None] * t + 2j * np.pi * k[:, None]).real')
    dct_basis = dft(size=40, phase=1).real
    b.plot(dct_basis)
with beat as b:
    print('# Decorrelate the signal using the DCT')
    x = x @ dct_basis.T
    print('x = x @ dct_basis.T')
    b.matshow(x)
with beat as b:
    print('# Take top 13 cepstral coefficients')
    print('x = x[..., :13]')
    x = x[..., :13]
    b.matshow(x)
with beat as b:
    print('# Zero mean')
    print('x = x - np.mean(x, axis=0, keepdims=True)')
    x = x - np.mean(x, axis=0, keepdims=True)
    b.matshow(x)
with beat as b:
    print('# Unit variance')
    print('x = x / np.std(x, axis=0, keepdims=True)')
    x = x / np.std(x, axis=0, keepdims=True)
    b.matshow(x)

beat.replay()
print('Done!')