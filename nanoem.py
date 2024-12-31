#%%
import matplotlib; matplotlib.use("kitcat")
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pathlib import Path
import numpy as np
import textwrap
from IPython.display import display
from tqdm import tqdm

from nanolib import *

frames = np.load('exp/frames.npy').astype(np.float32)
frames = cmvn(frames)
durations = np.load('exp/file_durations.npy')
transcript_tab = np.loadtxt('exp/transcripts.txt', dtype=str)
codebook = np.load('exp/codebook16384.npy')

#%%

def make_chain(state_sequence, id_weight=0.1):
    num_states = len(state_sequence)
    # allow transitions forward and self-loops
    chain = np.eye(num_states, k=1) + id_weight*np.eye(num_states)
    chain = chain / np.sum(chain, axis=1, keepdims=True)
    #chain[-1,-1] = 0.5
    return chain

example_id = np.where(transcript_tab[:, 0] == 'common_voice_uk_27626906')[0].item()
fig, (ax, axr) = plt.subplots(1, 2, figsize=(15, 5))
example = frames[:durations[example_id]]
ax.plot(example)
symbols = index_symbols(transcript_tab[:, 1])
lm = estimate_bigrams(transcript_tab[:, 1], symbols, bias=1)
label = str(transcript_tab[example_id, 1])
ends = durations1(label, durations[0])
draw_alignment(ax, ends, label)

path = 'wav/' + str(transcript_tab[example_id, 0]) + '.mp3'
audio = AudioSegment.from_mp3(path)
display(audio)

symbol_list = [symbol for symbol, _ in sorted(symbols.items(), key=lambda item: item[1])]

# label_lm = estimate_bigrams([label], symbols, bias=0.01)
# trans = label_lm + 0.2 * np.eye(len(label_lm))
# trans = trans / np.sum(trans, axis=1, keepdims=True)

#trans = np.float32(trans > 0.1)

state_repeats = 1
state_chain = [symbols[s] for s in label for rep in range(state_repeats)]
trans = make_chain(state_chain, id_weight=30*state_repeats)
print(trans)
#state_chain = list(range(len(symbol_list)))

im = display_lm(trans, symbols, ax=axr)
axr.set_title('transition')
plt.colorbar(im)
plt.close(fig)

pi_sim = np.triu(np.float32(np.array(state_chain)[None, :] == np.array(state_chain)[:, None]))
pi_sim = pi_sim / np.sum(pi_sim, axis=1, keepdims=True)

label_seq = []
for sym, dur in zip(label, np.append(ends[0], np.diff(ends))):
    label_seq.extend([symbols[sym]]*dur)
eps = 0.3
label_seq = (np.eye(len(symbols)) + eps)[label_seq]
label_seq = label_seq / np.sum(label_seq, axis=1,keepdims=True)
fig, ax = plt.subplots(1, 1, figsize=(17, 10))
ax.matshow(label_seq.T)
ax.set_yticks(ticks=np.arange(len(symbol_list)), labels=symbol_list, fontsize=8)
ax.set_title('prior')
plt.close(fig)

dev = example[None, :, :] - codebook[:, None, :]
example = example[60:160] # crop
prec = 1/np.mean(dev * dev, axis=1)
np.random.seed(42)
pi0 = np.ones((len(trans), 16384))
pi0 = pi0 / np.sum(pi0, axis=1, keepdims=True)

def dedup(x):
    x = np.char.array(x)
    mask = np.concatenate(([True], x[1:] != x[:-1]))
    return ''.join(x[mask])

#init = eps + np.eye(len(trans))[-1]
#init = init / np.sum(init)
init = trans[0]
pi = pi0
for step in range(16):
    comp_obs = logprob(example, codebook, prec, pi, agg=False, renormalize_weights=False)
    obs = np.exp(logsumexp(comp_obs))
    comp = np.exp(comp_obs) / obs[:, :, None] # component probabilities: nkm

    decoded = dedup([symbol_list[state_chain[i]] for i in decode(obs, init, trans)])
    #print('decoded', decoded)

    # state occupancy posterior
    loss, post, trans1 = state_posterior(obs, init, trans)
    if step % 1 == 0:
        fig, (ax, axr) = plt.subplots(1, 2, figsize=(24, 6))
        ax.matshow(post.T, aspect='auto')
        ax.set_xticks(ticks=range(0, 100, 5))
        ax.set_yticks(ticks=np.arange(len(label)*state_repeats), labels=''.join(l*state_repeats for l in label), fontsize=8)
        ax.set_title(f'{step=} log posterior for {label} {loss=:.07}')

        axr.matshow(trans1, aspect='auto')
        
        plt.show()
        plt.close(fig)

    #init = post[0]

    #post = post @ pi_sim.T # tie posteriors for similar states
    pi_c = np.sum(comp * post[:, :, None], axis=0)
    #pi_c = np.clip(pi_c, 1e-32, 1)
    #pi_c = pi_sim @ pi_c
    pi = pi_c / np.sum(post, axis=0)[:, None]
    pi = pi_sim @ pi

    print(np.sum(pi, axis=1))

plt.figure()
plt.matshow(obs.T)
plt.show()
print(decode(obs, init, trans))