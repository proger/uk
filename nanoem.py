#%%
try:
    import matplotlib; matplotlib.use("kitcat")
except ValueError:
    pass
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from pydub import AudioSegment
import numpy as np
from IPython.display import display

from nanolib import *

def dedup(x):
    x = np.char.array(x)
    mask = np.concatenate(([True], x[1:] != x[:-1]))
    return ''.join(x[mask])

frames = np.load('exp/frames.npy').astype(np.float32)
frames = cmvn(frames)
durations = np.load('exp/file_durations.npy')
transcript_tab = np.loadtxt('exp/transcripts.txt', dtype=str)
#codebook = np.load('exp/codebook16384.npy')
codebook = np.load('exp/codebook1024.npy')

np.random.seed(32)
frame_permutation = np.random.permutation(len(frames))
train = frames[frame_permutation[:10000]]
precision = 1/np.mean((train[None, :, :] - codebook[:, None, :])**2, axis=1)

def make_chain(state_sequence, num_frames):
    num_states = len(state_sequence)
    id_weight = -num_states/num_frames + 1
    # allow transitions forward and self-loops
    chain = (1-id_weight)*np.eye(num_states, k=1) + id_weight*np.eye(num_states)
    chain[-1, -1] = 1 # terminal state
    return chain

#%%

example_id = np.where(transcript_tab[:, 0] == 'common_voice_uk_27626906')[0].item()
cumulative_durations = np.cumsum(durations)
example = frames[cumulative_durations[example_id-1]:cumulative_durations[example_id]]
symbols = index_symbols(transcript_tab[:, 1])
label = str(transcript_tab[example_id, 1])

path = 'wav/' + str(transcript_tab[example_id, 0]) + '.mp3'
audio = AudioSegment.from_mp3(path)
#display(audio)

symbol_list = [symbol for symbol, _ in sorted(symbols.items(), key=lambda item: item[1])]

state_repeats = 1
label_with_repeats = ''.join([l*state_repeats for l in label])
state_chain = [symbols[s] for s in label for rep in range(state_repeats)]
trans = make_chain(state_chain, len(example))

pi_sim = np.triu(np.float32(np.array(state_chain)[None, :] == np.array(state_chain)[:, None]))
pi_sim = pi_sim / np.sum(pi_sim, axis=1, keepdims=True)

# codebook = example
# precision = 1/np.mean((codebook[None, :, :] - codebook[:, None, :])**2, axis=1)

pi0 = np.ones((len(trans), len(codebook))) / len(codebook)

init = np.eye(len(trans))[0]
pi = pi0
for step in range(30):
    comp = logprob(example, codebook, precision, pi, agg=False, renormalize_weights=False) # component logits: nkm
    obs = np.exp(logsumexp(comp)) # mixture probability: nk
    p_comp = np.exp(comp) # component probabilities
    response = p_comp / obs[:, :, None] # component responsibilities: nkm

    decoded = dedup([symbol_list[state_chain[i]] for i in decode(obs, init, trans)])
    #print('decoded', decoded)

    # state occupancy posterior
    loss, post, trans1, alpha = state_posterior(obs, init, trans)
    if step % 1 == 0:
        fig, (ax, axr) = plt.subplots(1, 2, figsize=(24, 6))
        #ax.matshow(post.T, aspect='auto')
        ax.matshow(alpha.T, aspect='auto')
        #ax.set_xticks(ticks=gt_ends)
        ax.set_yticks(ticks=np.arange(len(label)*state_repeats), labels=''.join(l*state_repeats for l in label), fontsize=14)
        #ax.set_title(f'{step=} posterior for {label} {loss=:.07}')
        ax.set_title(f'{step=} forward variables for {label} {loss=:.07}')

        axr.set_xticks(ticks=np.arange(len(label)*state_repeats), labels=''.join(l*state_repeats for l in label), fontsize=14)
        axr.set_yticks(ticks=np.arange(len(label)*state_repeats), labels=''.join(l*state_repeats for l in label), fontsize=14)
        axr.matshow(trans1, aspect='auto')
        axr.set_title('transition posterior')
        
        plt.show()
        plt.close(fig)

    pi_c = np.sum(response * post[:, :, None] , axis=0)
    #pi_c = np.clip(pi_c, 1e-32, 1)
    pi = pi_c / np.sum(post, axis=0)[:, None]
    pi = pi_sim @ pi
    print(-np.sum(pi * np.log(pi), axis=1), 'mixture entropies')

    #print(np.sum(pi, axis=1), 'pi sums must be ones')

fig, ax = plt.subplots(1, 1, figsize=(24, 6))
ax.matshow(example.T, aspect='auto')
states = decode(obs, init, trans)
ali = np.cumsum(np.unique(states, return_counts=True)[1])
draw_alignment(ali, label_with_repeats, ax=ax)
ax.set_xticks([])
print('state durations:', ali)
plt.show()
plt.close(fig)
