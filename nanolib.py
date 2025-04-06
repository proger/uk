from pathlib import Path
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
from IPython.display import display
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
SPACE = '␣'

SAMPLING_RATE = 16000
FPS = 100
WINDOW_SIZE = 320
HOP_SIZE = 160

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

def encode_text(s):
    return SPACE + SPACE.join(s.split(' ')) + SPACE

def read_table(file_path, skip=0, max_length=10000):
    entries = {}
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i < skip:
                continue
            key, text = line.strip().split(maxsplit=1)
            text = encode_text(text)
            if len(text) < max_length:
                entries[key] = text
    return entries

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

def mfcc(waveform):
    window = hann_window()
    dft_basis = dft()
    mel_basis = warp(hz_to_mel, mel_to_hz, 40, 320, a=50, b=16000)
    dct_basis = dft(size=40, phase=1).real

    x = preemph(waveform)
    x = np.lib.stride_tricks.sliding_window_view(x, WINDOW_SIZE)[::HOP_SIZE]
    x = window * x
    x = x @ dft_basis.T
    x = np.abs(x)**2
    x = x @ mel_basis.T
    x = np.log10(x + 1e-2)
    x = x @ dct_basis.T
    x = x[..., :13]
    return x

def cmvn(frames):
    frames = frames - np.mean(frames, axis=0, keepdims=True)
    frames = frames / np.std(frames, axis=0, keepdims=True)
    return frames

def extract_mfcc(key, wav_dir=Path('wav')):
    audio = load(Path(key) if Path(key).exists() else wav_dir / f'{key}.mp3')
    waveform = to_samples(audio)
    x = mfcc(waveform)
    return x

def index_symbols(lines):
    return {c: i for i, c in enumerate(sorted(set([c for line in lines for c in line])))}

def estimate_bigrams(lines, symbols, bias=0):
    # from uk.clean_text import keep_useful_characters
    # lines = list(filter(None, map(keep_useful_characters, lines)))

    vocab_size = len(symbols)
    counts = np.zeros((vocab_size, vocab_size)) + bias
    for line in lines:
        for source, target in zip(line, line[1:]):
            counts[symbols[source],symbols[target]] += 1
    # normalize rows to get valid p(target|source)
    counts = counts / np.sum(counts, axis=1, keepdims=True)
    return counts

def draw_alignment(durations, labels, ax=None, yloc=-1.1):
    if ax is None:
        ax = plt.gca()
    start = 0
    for i, duration in enumerate(durations):
        ax.axvline(duration, alpha=1, color='red', lw=3)
        length = duration - start
        #ax.text(start + length/2, yloc, labels[i])
        ax.text(start, yloc, labels[i])
        start = duration

def stoch_round(x):
    x = np.asarray(x)
    integer_parts = np.floor(x)
    fractional_parts = x - integer_parts
    random_probs = np.random.rand(*x.shape)
    rounded_result = np.where(random_probs < fractional_parts, np.ceil(x), np.floor(x))
    return rounded_result.astype(int)

def durations1(labels, duration):
    return np.cumsum(stoch_round([duration / len(labels)]*len(labels)))

def logsumexp(x, axis=-1, keepdims=False):
    max = np.max(x, axis=axis, keepdims=keepdims)
    x = np.exp(x - (max[..., None] if keepdims == False else max))
    return np.log(np.sum(x, axis=axis, keepdims=keepdims)) + max

def logprob(x, mean, precision, weights=None, renormalize_weights=True, agg=True):
    "nd,kd,kd,mk->nm"
    d = precision.shape[-1]
    z = x[:, None, :] - mean
    logdet = -np.sum(np.log(precision), axis=-1)
    k = d * np.log(2*np.pi)
    log_prob = -0.5 * (k + logdet + np.einsum("nkd,kd,nkd->nk", z, precision, z))
    if weights is None:
        return log_prob # nk
    else:
        if renormalize_weights:
            weights = weights / np.sum(weights, axis=-1, keepdims=True)
        if agg:
            return logsumexp(log_prob[:, None, :] + np.log(weights)) # nm
        else:
            return log_prob[:, None, :] + np.log(weights) # nmk

def decode(obs, init, trans):
    "tm,m,mk->t"
    T = obs.shape[0]
    back = np.zeros_like(obs, dtype=int)
    delta = np.zeros_like(obs)
    delta[0, :] = init * obs[0]
    scales = np.ones((T, 1))
    scales[0] = np.sum(delta[0, :])
    delta[0, :] /= scales[0]

    for t in range(1, T):
        trans_t = delta[t-1, :, None] * trans
        back[t, :] = np.argmax(trans_t, axis=0)
        delta[t, :] = np.max(trans_t, axis=0) * obs[t, :]

        scales[t] = np.sum(delta[t, :])
        if scales[t] == 0:
            scales[t] = 1
        delta[t, :] /= scales[t]

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1, :])

    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]

    return path


def forward(obs, init, trans):
    "tm,m,mk->tm. compute p(state_t | obs<t)"
    T = obs.shape[0]
    alpha = np.zeros_like(obs)
    alpha[0, :] = init * obs[0]
    c = np.ones((T, 1))
    c[0] = np.sum(alpha[0])
    ahat = alpha / c

    for t in range(1, T):
        trans_t = np.einsum('m,mk->mk', ahat[t-1, :], trans)
        alpha[t, :] = np.einsum('mk,k->k', trans_t, obs[t, :])
        c[t] = np.sum(alpha[t])
        if c[t] == 0:
            c[t] = 1
        ahat[t] = alpha[t] / c[t]

    return ahat, c


def backward(obs, transT):
    "tm,km->tm"
    T = obs.shape[0]
    beta = np.ones_like(obs)
    c = np.ones((T, 1))
    c[T-1, :] = np.sum(beta[T-1])
    bhat = beta / c

    for t in range(T-2, -1, -1):
        trans_t = np.einsum('m,mk->mk', bhat[t+1, :], transT)
        beta[t, :] = np.einsum('m,mk->k', obs[t+1, :], trans_t)
        c[t] = np.sum(beta[t])
        if c[t] == 0:
            c[t] = 1
        bhat[t] = beta[t] / c[t]

    return bhat, c


def state_posterior(obs, init, trans):
    T = obs.shape[0]
    alpha, ca = forward(obs, init, trans)
    beta, cb = backward(obs, trans)
    p = alpha * beta

    p = np.clip(p, 1e-32, 1.0)

    state_post = p / np.sum(p, axis=-1, keepdims=True) # occupancy posterior

    # transition posterior
    trans_post = np.zeros_like(trans)
    for t in range(T):
        if t == T-1:
            b = 1
        else:
            b = obs[t+1, None, :] * beta[t+1, None, :]
        update = alpha[t, :, None] * trans * b
        trans_post += update / np.sum(update)

    sum_state_post = np.sum(state_post, axis=0)
    new_trans = trans_post / sum_state_post[:, None]

    log_loss = -np.sum(np.log(ca))
    
    return log_loss, state_post, new_trans, alpha


def test_state_posterior():
    # example from https://en.wikipedia.org/wiki/Forward–backward_algorithm
    trans = np.array([[0.7, 0.3], [0.3, 0.7]])
    umb = np.diag(np.array([[0.9, 0.0], [0.0, 0.2]]))
    noumb = np.diag(np.array([[0.1, 0.0], [0.0, 0.8]]))
    init = np.array([0.5, 0.5])
    obs = np.stack([umb, umb, noumb, umb, umb])
    log_loss, post, new_trans = state_posterior(obs, init, trans)

    assert np.allclose(post, np.array([[0.86733889, 0.13266111],
                                       [0.82041905, 0.17958095],
                                       [0.30748358, 0.69251642],
                                       [0.82041905, 0.17958095],
                                       [0.86733889, 0.13266111]]))

    print(log_loss, 'loss')
    print(trans, 'old trans')
    print(new_trans, 'new trans')
    print(decode(obs, init, trans))

def display_lm(lm, symbols, ax=None):
    if ax is None:
        ax = plt.gca()

    im = ax.matshow(lm)
    symbol_list = [symbol for symbol, _ in sorted(symbols.items(), key=lambda item: item[1])]

    ax.set_xticks(ticks=np.arange(len(symbol_list)), labels=symbol_list, rotation=90)
    ax.set_yticks(ticks=np.arange(len(symbol_list)), labels=symbol_list)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.set_xlabel('target', fontsize=10)
    ax.set_ylabel('source', fontsize=10)
    ax.xaxis.set_label_position('top')
    return im

def l2_attend(q, k):
    # https://arxiv.org/abs/2006.04710 appendix E
    qsqnorm = np.sum(q**2, axis=1)[:, None]
    ksqnorm = np.sum(k**2, axis=1)[None, :]

    att2 = qsqnorm + ksqnorm - 2 * np.dot(q, k.T)
    att2 = np.maximum(att2, 0)
    return np.sqrt(att2)

def vq_loss(codebook, examples):
    att = l2_attend(examples, codebook)
    loss = np.mean(np.min(att, axis=1))
    util = len(np.unique(np.argmin(att, axis=1)))
    return loss, util

def kmeans(codebook, train, eval, max_steps=20, batch_size=32768, lr=0.1, revive_below=0):
    codebook = np.copy(codebook)
    loss, util = vq_loss(codebook, eval)
    eval_losses = [loss]
    eval_utils = [util]
    train_util = np.ones(len(codebook))
    train_p = train_util / train_util.sum()
    train_ent = -np.sum(train_p * np.log(train_p))
    print('loss util entropy')
    print(loss, util, train_ent)

    for step in range(max_steps):
        batch = train[np.random.choice(len(train), batch_size)]
        att = l2_attend(batch, codebook)
        labels = np.argmin(att, axis=1)

        for i in range(len(codebook)):
            examples = batch[labels == i]
            centroid = np.mean(examples, axis=0)
            if not np.any(np.isnan(centroid)):
                codebook[i] = (1 - lr) * codebook[i] + lr * centroid
                train_util[i] += 1

        train_p = train_util / train_util.sum()
        train_ent = -np.sum(train_p * np.log(train_p))
        #print(train_util)

        loss, util = vq_loss(codebook, eval)
        eval_losses.append(loss)
        eval_utils.append(util)

        if step % 10 == 8 and revive_below:
            dead = train_util < revive_below
            num_dead = np.sum(dead)
            codebook[dead] = train[np.random.choice(len(train), num_dead)]
            print(loss, util, train_ent, 'revived', num_dead)
        else:
            print(loss, util, train_ent)
 
    return eval_losses, eval_utils, train_util, codebook


def lbg(train, eval, num_clusters=16384):
    codebook = np.mean(train, axis=0)[None, :]
    k = 1
    loss, util = vq_loss(codebook, eval)
    losses = [loss]
    utils = [util]
    print(k, 'init loss', loss, 'util', util)

    while k < num_clusters:
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        ax.matshow(codebook.T, aspect='auto')
        plt.show()
        plt.close(fig)

        perturbed_codebook = np.vstack([
            codebook * 1.1,
            codebook * 0.9
        ])

        loss = vq_loss(perturbed_codebook, eval)
        refine_losses, refine_utils, train_util, codebook = kmeans(perturbed_codebook, train, eval)
        losses.extend(refine_losses)
        utils.extend(refine_utils)
        k = len(codebook)
        print('k', k, 'refined loss', losses[-1], 'util', utils[-1])
    return losses, utils, codebook


if __name__ == '__main__':
    test_state_posterior()
