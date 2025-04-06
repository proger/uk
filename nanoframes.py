import argparse
try:
    import matplotlib; matplotlib.use("kitcat")
except ValueError:
    pass
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from pathlib import Path
import numpy as np
from nanolib import *
from tqdm import tqdm

def main(num_clusters, max_steps=30, batch_size=32768):
    # Read and process transcripts
    transcripts = read_table(Path('data/mozilla-foundation/common_voice_10_0/uk/train/text'))
    train_keys = list(transcripts.keys())

    if not Path('exp/frames.npy').exists():
        # Process segments
        segments = [extract_mfcc(key) for key in tqdm(train_keys)]
        #segments = thread_map(extract_mfcc, train_keys)
        frames = np.concatenate(segments, axis=0).astype(np.float32)
        durations = np.array([x.shape[0] for x in segments])

        # Plot example histogram
        fig, ax = plt.subplots(1, 1)
        ax.hist(durations)
        ax.set_title('Examples')
        ax.set_xlabel('Frame Durations')
        ax.set_ylabel('Count')
        plt.show()
        plt.close(fig)

        # Save processed data
        np.save('exp/frames.npy', frames)
        np.save('exp/file_durations.npy', durations)
        np.savetxt('exp/transcripts.txt', np.array(list(transcripts.items())), fmt='%s')

    # Load and preprocess data
    frames = np.load('exp/frames.npy').astype(np.float32)
    frames = cmvn(frames)
    durations = np.load('exp/file_durations.npy')
    transcript_tab = np.loadtxt('exp/transcripts.txt', dtype=str)

    np.random.seed(32)
    perm = np.random.permutation(len(frames))
    train, eval = frames[perm[:-10000]], frames[perm[-10000:]]
    best_seed, best_loss = -1, float('inf')
    rand_losses, rand_utils = [], []

    lbg_losses, lbg_utils, codebook = lbg(train, eval, num_clusters)
    np.save(f'exp/codebook{num_clusters}.npy', codebook)
    print(f'exp/codebook{num_clusters}.npy')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain the acoustic model.')
    parser.add_argument('--num_clusters', type=int, default=1024, help='Number of clusters for LBG.')
    parser.add_argument('--max_steps', type=int, default=30, help='Maximum steps for k-means algorithm.')
    parser.add_argument('--batch_size', type=int, default=32768, help='Batch size for attention analysis.')
    args = parser.parse_args()

    main(args.num_clusters, args.max_steps, args.batch_size)
