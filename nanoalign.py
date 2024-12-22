#%%
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pathlib import Path
import numpy as np
import textwrap
from IPython.display import display
from tqdm import tqdm

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
SPACE = '␣'


def load(file_path):
    audio = AudioSegment.from_mp3(file_path)
    audio = audio.set_frame_rate(16000)
    return audio


def to_samples(audio, normalize=True):
    samples = np.array(audio.get_array_of_samples())
    range_max = 2**(audio.sample_width*8-1)
    samples = samples.astype(np.float32) / range_max
    if normalize:
        samples = samples / np.max(samples)
    return samples


def read_table(file_path, skip=0, max_length=10000, rewrite_text=lambda x: x):
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


def stoch_round(x):
    x = np.asarray(x)
    integer_parts = np.floor(x)
    fractional_parts = x - integer_parts
    random_probs = np.random.rand(*x.shape)
    rounded_result = np.where(random_probs < fractional_parts, np.ceil(x), np.floor(x))
    return rounded_result.astype(int)


def durations1(labels, duration):
    return np.cumsum(stoch_round([duration / len(labels)]*len(labels)))


def draw_alignment(ax, durations, labels):    
    start = 0
    for i, duration in enumerate(durations):
        ax.axvline(duration, alpha=0.5)
        length = duration - start
        ax.text(start + length/2, -1.1, labels[i])
        start = duration


def display_segmentation(audio, ends):
    start = 0
    for i, end in enumerate(ends):
        display(audio.get_sample_slice(start,end))
        start = end


def rewrite_text(s):
    return SPACE + SPACE.join(s.split(' ')) + SPACE

def index_symbols(lines):
    return {c: i for i, c in enumerate(sorted(set([c for line in lines for c in line])))}


def estimate_bigrams(lines, symbols):
    vocab_size = len(symbols)
    counts = np.zeros((vocab_size, vocab_size))
    for line in lines:
        for source, target in zip(line, line[1:]):
            counts[symbols[source],symbols[target]] += 1
    # normalize rows to get valid p(target|source)
    counts = counts / np.sum(counts, axis=1, keepdims=True)
    return counts


def display_lm(ax, lm, symbols):
    ax.matshow(lm)
    symbol_list = [symbol for symbol, _ in sorted(symbols.items(), key=lambda item: item[1])]

    ax.set_xticks(ticks=np.arange(len(symbol_list)), labels=symbol_list, rotation=90)
    ax.set_yticks(ticks=np.arange(len(symbol_list)), labels=symbol_list)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.set_xlabel('target', fontsize=10)
    ax.set_ylabel('source', fontsize=10)
    ax.xaxis.set_label_position('top')


wav_dir = Path('wav')
transcripts = read_table(Path('data/mozilla-foundation/common_voice_10_0/uk/train/text'), skip=20, max_length=30, rewrite_text=rewrite_text)
lexicon = read_table(Path('data/local/dict/lexicon_common_voice_uk.txt'))

lines = [transcripts[stem] for stem in transcripts]
symbols = index_symbols(lines)
lm = estimate_bigrams(lines, symbols)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
display_lm(ax, lm, symbols)

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]


def collect_examples(audio, ends, label, acc):
    start = 0
    for i, end in enumerate(ends):
        acc[label[i]].append(audio.get_sample_slice(start, end))
        start = end


for ax, key in zip(axes, transcripts):
    audio = load(wav_dir / f'{key}.mp3')
    waveform = to_samples(audio)
    label = transcripts[key]
    print(key, label)
    
    ax.set_title('\n'.join(textwrap.wrap(label, width=40)))
    ax.plot(waveform)
    ax.set_xlim(0)
    ends = durations1(label, len(waveform))
    draw_alignment(ax, ends, label)

plt.tight_layout()
plt.show()

print('estimating model: pass 1, uniform alignment')

acc = {sym: [] for sym in symbols}

for key in tqdm(transcripts):
    audio = load(wav_dir / f'{key}.mp3')
    waveform = to_samples(audio)
    label = transcripts[key]
    ends = durations1(label, len(waveform))
    collect_examples(audio, ends, label, acc)

a = AudioSegment.empty()
for seg in acc['а']:
    a += seg
plt.plot(to_samples(a))
a

#%%

SAMPLING_RATE = 16000
FPS = 100
WINDOW_SIZE = 320
HOP_SIZE = 160


def preemph(x, c=0.97):
    return np.append(x[0], x[1:] - c * x[:-1])

def hann_window(size=WINDOW_SIZE):
    n = np.arange(size)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (size - 1)))

def dft(size=WINDOW_SIZE, rate=16000, min_freq=50):
    k = np.linspace(min_freq, rate / 2, size, endpoint=False)
    t = np.arange(size) / rate
    return np.exp(-2j * np.pi * k[:, None] * t)

def dct(size=40):
    k = np.arange(size) / (2 * size)
    t = np.arange(size)
    return np.cos(-2 * np.pi * k[:, None] * t + 2 * np.pi * k[:, None])

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def mel_filter_bank(out_channels=40, in_channels=320, min_hz=50, max_hz=16000, sample_rate=16000):
    mel_points = np.linspace(hz_to_mel(min_hz), hz_to_mel(max_hz), out_channels + 2)
    hz_points = mel_to_hz(mel_points)
    
    bin_points = np.floor(in_channels * hz_points / sample_rate).astype(int)
    
    filters = np.zeros((out_channels, in_channels))
    
    for i in range(1, out_channels + 1):
        filters[i - 1, bin_points[i - 1]:bin_points[i]] = (
            (np.arange(bin_points[i - 1], bin_points[i]) - bin_points[i - 1]) /
            (bin_points[i] - bin_points[i - 1])
        )
        filters[i - 1, bin_points[i]:bin_points[i + 1]] = (
            (bin_points[i + 1] - np.arange(bin_points[i], bin_points[i + 1])) /
            (bin_points[i + 1] - bin_points[i])
        )
    
    return filters

window = hann_window()
dft_basis = dft()
mel = mel_filter_bank()
dct_basis = dct()
for row in mel:
    print(np.sum(row))
plt.matshow(mel)
print(mel.shape, 'mel')

plt.figure()

waves = preemph(waveform)
waves = np.lib.stride_tricks.sliding_window_view(waves, WINDOW_SIZE)[::HOP_SIZE]
frames = []
for frame in waves:
    x = window * frame
    x = dft_basis @ x
    x = np.abs(x)**2
    x = mel @ x
    x = np.log10(x + 1e-2)
    x = dct_basis @ x
    x = x[:13]
    plt.plot(x)
    frames.append(x)
plt.show()
plt.matshow(np.array(frames).T[:,30:200], origin='lower')
