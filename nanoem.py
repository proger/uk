#%%
import argparse
try:
    import matplotlib; matplotlib.use("kitcat")
except ValueError:
    pass
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import numpy as np

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
        return argparse.Namespace(key=['common_voice_uk_27626906'], label=None, uber=None, steps=10, show=True, codebook_size=512)

    parser = argparse.ArgumentParser(description="Process audio and label for the example.")
    parser.add_argument('--uber', type=str, help='Path to the pretrained uber_pi file.')
    parser.add_argument('--label', type=str)
    parser.add_argument('--key', type=str, nargs='+', default=['common_voice_uk_27626906'])
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('-t', '--time_factor', type=float, default=1, help='make_chain duration factor')
    parser.add_argument('-k', '--codebook_size', type=int, help='Codebook size', default=512)
    parser.add_argument('-o', '--output', type=str, help='Path to the output file.')
    return parser.parse_args()

#%%

args = parse_args()
frames = np.load('exp/frames.npy').astype(np.float32)
frames = cmvn(frames)
codebook = np.load(f'exp/codebook{args.codebook_size}.npy')
cumulative_durations = np.cumsum(np.load('exp/file_durations.npy'))
transcript_tab = np.loadtxt('exp/transcripts.txt', dtype=str)
symbols = index_symbols(transcript_tab[:, 1])
np.random.seed(32)
frame_permutation = np.random.permutation(len(frames))
train = frames[frame_permutation[:10000]]
precision = 1/np.mean((train[None, :, :] - codebook[:, None, :])**2, axis=1)
symbol_list = [symbol for symbol, _ in sorted(symbols.items(), key=lambda item: item[1])]
print('symbols', symbol_list)

def take_example(key, state_repeats=1, _cache={}):
    if key in _cache:
        return _cache[key]

    if Path(key).exists():
        path = args.path
        label = encode_text(args.label)
        example = cmvn(extract_mfcc(path))
    else:
        example_id = np.where(transcript_tab[:, 0] == key)[0].item()
        label = str(transcript_tab[example_id, 1])
        path = 'wav/' + str(transcript_tab[example_id, 0]) + '.mp3'
        example = frames[cumulative_durations[example_id-1]:cumulative_durations[example_id]]

    state_repeats = 1
    label = ''.join([l*state_repeats for l in label])
    state_chain = [symbols[s] for s in label for rep in range(state_repeats)]
    trans = make_chain(state_chain, len(example)*args.time_factor)
    print(trans, 'transition matrix')
    init = np.eye(len(trans))[0]

    pi_sim = np.triu(np.float32(np.array(state_chain)[None, :] == np.array(state_chain)[:, None]))
    pi_sim = pi_sim / np.sum(pi_sim, axis=1, keepdims=True)

    uber_to_local = np.eye(len(symbols))[state_chain]
    #print(uber_to_local, state_chain)
    uber_to_local = uber_to_local / np.sum(uber_to_local, axis=1, keepdims=True)

    _cache[key] = (example, init, trans, state_chain, pi_sim, label, uber_to_local)
    return _cache[key]


if args.uber:
    uber_pi = np.load(args.uber)
else:
    uber_pi = 10 * np.random.rand(len(symbols), len(codebook))
    uber_pi = uber_pi / np.sum(uber_pi, axis=1, keepdims=True)

for step in range(args.steps):
    update = np.zeros_like(uber_pi)
    agg_loss = 0

    for key in args.key:
        example, init, trans, state_chain, pi_sim, label, uber_to_local = take_example(key)

        local_pi = uber_to_local @ uber_pi

        comp = logprob(example, codebook, precision, local_pi, agg=False, renormalize_weights=False) # component logits: nkm
        obs_logits = logsumexp(comp) # mixture logits: nk
        response = np.exp(comp - obs_logits[:, :, None]) # softmaxed component responsibilities: nkm
        obs = np.exp(obs_logits)

        # state occupancy posterior
        loss, post, trans1, alpha = state_posterior(obs, init, trans)
        if args.show and step % 1 == 0:
            decoded = dedup([symbol_list[state_chain[i]] for i in decode(obs, init, trans)])
            #print('decoded', decoded)

            states = decode(obs, init, trans)
            ali = np.cumsum(np.unique(states, return_counts=True)[1])

            fig, (ax, axr, axa, axo) = plt.subplots(1, 4, figsize=(24, 6))
            #ax.matshow(post.T, aspect='auto')
            ax.matshow(alpha.T, aspect='auto')
            ax.set_yticks(ticks=np.arange(len(label)), labels=label, fontsize=14)
            ax.set_title(f'{step=} forward for {label} {loss=:.07}')

            axr.set_xticks(ticks=np.arange(len(label)), labels=label, fontsize=14)
            axr.set_yticks(ticks=np.arange(len(label)), labels=label, fontsize=14)
            axr.matshow(np.log(trans1 + 1e-12), aspect='auto')
            axr.set_title('trans')

            axa.matshow(example.T, aspect='auto')
            draw_alignment(ali, label, ax=axa, yloc=1.0)
            axa.set_title(f'ali {ali}')

            axo.matshow(np.log(obs.T), aspect='auto')
            axo.set_yticks(ticks=np.arange(len(label)), labels=label, fontsize=14)
            draw_alignment(ali, label, ax=axo, yloc=1.0)
            axo.set_title('obs')

            plt.tight_layout()
            plt.show()
            plt.close(fig)
        #if step > 30:
        #    trans = 0.99 * trans + 0.01 * trans1

        local_pi_c = np.sum(response * post[:, :, None] , axis=0)
        local_pi1 = local_pi_c / np.sum(post, axis=0)[:, None]
        local_pi1 = pi_sim @ local_pi1 # redistribute between common symbols in a sequence
        #print(key, -np.sum(local_pi1 * np.log(local_pi1), axis=1), 'mixture entropies', 'step', step)

        update += uber_to_local.T @ local_pi1
        agg_loss += loss

        assert np.allclose(np.sum(local_pi1, axis=1), 1)

    uber_pi = update / len(args.key)
    loss = agg_loss / len(args.key)
    print('step', step, 'loss', loss)

if args.output:
    np.save(args.output, uber_pi)
