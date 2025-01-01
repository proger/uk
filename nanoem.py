#%%
import argparse
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

def make_chain(state_sequence, num_frames):
    num_states = len(state_sequence)
    id_weight = -num_states/num_frames + 1
    # allow transitions forward and self-loops
    chain = (1-id_weight)*np.eye(num_states, k=1) + id_weight*np.eye(num_states)
    chain[-1, -1] = 1 # terminal state
    return chain

def is_jupyter():
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False

def parse_args():
    if is_jupyter():
        return argparse.Namespace(label='common_voice_uk_27626906', path=None, uber=None)

    parser = argparse.ArgumentParser(description="Process audio and label for the example.")
    parser.add_argument('--uber', type=str, help='Path to the uber_pi file.')
    parser.add_argument('--path', type=str, help='Path to the MP3 file.')
    parser.add_argument('--label', type=str, default='common_voice_uk_27626906',
                        help='Label text.')
    return parser.parse_args()

#%%

args = parse_args()
frames = np.load('exp/frames.npy').astype(np.float32)
frames = cmvn(frames)
#codebook = np.load('exp/codebook16384.npy')
codebook = np.load('exp/codebook1024.npy')
transcript_tab = np.loadtxt('exp/transcripts.txt', dtype=str)
symbols = index_symbols(transcript_tab[:, 1])

if not args.path:
    durations = np.load('exp/file_durations.npy')

    example_id = np.where(transcript_tab[:, 0] == args.label)[0].item()
    cumulative_durations = np.cumsum(durations)
    label = str(transcript_tab[example_id, 1])
    path = 'wav/' + str(transcript_tab[example_id, 0]) + '.mp3'
    audio = load(path)
    example = frames[cumulative_durations[example_id-1]:cumulative_durations[example_id]]
else:
    path = args.path
    label = encode_text(args.label)
    audio = AudioSegment.from_mp3(path)
    example = cmvn(extract_mfcc(path))

#display(audio)

np.random.seed(32)
frame_permutation = np.random.permutation(len(frames))
train = frames[frame_permutation[:10000]]
precision = 1/np.mean((train[None, :, :] - codebook[:, None, :])**2, axis=1)

symbol_list = [symbol for symbol, _ in sorted(symbols.items(), key=lambda item: item[1])]

state_repeats = 1
label_with_repeats = ''.join([l*state_repeats for l in label])
state_chain = [symbols[s] for s in label for rep in range(state_repeats)]
trans = make_chain(state_chain, len(example))

if args.uber:
    uber_pi = np.load(args.uber)
else:
    uber_pi = np.ones((len(symbols), len(codebook))) / len(codebook)
uber_to_local = np.triu(np.eye(len(symbols))[state_chain])
uber_to_local = uber_to_local / np.sum(uber_to_local, axis=1, keepdims=True)

pi_sim = np.triu(np.float32(np.array(state_chain)[None, :] == np.array(state_chain)[:, None]))
pi_sim = pi_sim / np.sum(pi_sim, axis=1, keepdims=True)

# codebook = example
# precision = 1/np.mean((codebook[None, :, :] - codebook[:, None, :])**2, axis=1)

init = np.eye(len(trans))[0]
for step in range(40):
    local_pi = uber_to_local @ uber_pi

    comp = logprob(example, codebook, precision, local_pi, agg=False, renormalize_weights=False) # component logits: nkm
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
    if step > 30:
        trans = 0.99 * trans + 0.01 * trans1

    local_pi_c = np.sum(response * post[:, :, None] , axis=0)
    #pi_c = np.clip(pi_c, 1e-32, 1)
    local_pi1 = local_pi_c / np.sum(post, axis=0)[:, None]
    local_pi1 = pi_sim @ local_pi1 # redistribute between common symbols in a sequence
    print(-np.sum(local_pi1 * np.log(local_pi1), axis=1), 'mixture entropies')

    uber_pi = 0.9 * uber_pi + 0.1 * (uber_to_local.T @ local_pi1)

    #print(np.sum(pi, axis=1), 'pi sums must be ones')

    if step % 10 == 9 or step == 0:
        fig, ax = plt.subplots(1, 1, figsize=(24, 6))
        ax.matshow(example.T, aspect='auto')
        states = decode(obs, init, trans)
        ali = np.cumsum(np.unique(states, return_counts=True)[1])
        draw_alignment(ali, label_with_repeats, ax=ax)
        ax.set_xticks([])
        print('state durations:', ali)
        plt.show()
        plt.close(fig)

np.save('exp/uber_pi.npy', uber_pi)
