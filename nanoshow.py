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


def parse_args():
    parser = argparse.ArgumentParser(description="Show the file")
    parser.add_argument('filename', type=str, help='Path to the numpy.')
    return parser.parse_args()

args = parse_args()
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.matshow(np.load(args.filename), aspect='auto')
plt.show()
