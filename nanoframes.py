#%%
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from nanolib import *


transcripts = read_table(Path('data/mozilla-foundation/common_voice_10_0/uk/train/text'))
train_keys = list(transcripts.keys())

#%%
#segments = thread_map(extract_mfcc, train_keys)
segments = [extract_mfcc(key) for key in tqdm(train_keys)]
#with ThreadPoolExecutor() as executor:
#    segments = list(executor.map(extract_mfcc, train_keys))
frames = np.concatenate(segments, axis=0).astype(np.float32)
durations = np.array([x.shape[0] for x in segments])

#%%
fig, ax = plt.subplots(1, 1)
ax.hist(durations)
ax.set_title('examples')
ax.set_xlabel('frame durations')
ax.set_ylabel('count')

#%%
np.save('exp/frames.npy', frames)
np.save('exp/file_durations.npy', durations)
np.savetxt('exp/transcripts.txt', np.array(list(transcripts.items())), fmt='%s')

#%%
frames = np.load('exp/frames.npy').astype(np.float32)
frames = cmvn(frames)
durations = np.load('exp/file_durations.npy')
transcript_tab = np.loadtxt('exp/transcripts.txt', dtype=str)


np.random.seed(32)
perm = np.random.permutation(len(frames))
train, eval = frames[perm[:-10000]], frames[perm[-10000:]]
num_clusters = 1024
best_seed = 34

#%%
best_seed, best_loss = -1, float('inf')
rand_losses, rand_utils = [], []
for i in range(50):
    seed = 33 + i
    np.random.seed(seed)
    codebook = train[np.random.choice(len(train), num_clusters)]
    loss, util = vq_loss(codebook, eval)
    if loss < best_loss:
        best_seed, best_loss = seed, loss
    rand_losses.append(best_loss)
    rand_utils.append(util)
    print(loss)

best_seed

#%%

np.random.seed(best_seed)
codebook = train[np.random.choice(len(train), num_clusters)]

km_losses, km_utils, train_util, codebook = kmeans(codebook, train, eval, max_steps=30)
plt.plot(rand_losses, label='random search')
plt.plot(km_losses, label='lloyd (k means)')
plt.legend()


#%%
plt.matshow(frames.T, aspect='auto')

lbg_losses, lbg_utils, codebook = lbg(train, eval, num_clusters)

#%%
plt.plot(rand_losses, label='random search')
plt.plot(km_losses, label='minibatch k means')
plt.plot(lbg_losses, label='lbg 1.1, 0.9')
plt.yscale('log')
plt.legend()

#%%
np.save(f'exp/codebook{num_clusters}.npy', codebook)

#%%

codebook = np.load('exp/codebook16384.npy')
codebook.shape

batch_size = 32768
batch = train[np.random.choice(len(train), batch_size)]
att = l2_attend(batch, codebook)
plt.matshow(att)
#%%
len(np.unique(np.argmin(att, axis=1)))