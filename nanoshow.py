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

symbols = ["'", 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я', 'є', 'і', 'ґ', '␣']

args = parse_args()
fig, ax = plt.subplots(1, 1, figsize=(30, 20))
im = ax.matshow(np.log(np.load(args.filename)), aspect='auto', vmin=-100)
#im = ax.matshow(np.load(args.filename), aspect='auto')
ax.set_yticks(ticks=np.arange(len(symbols)), labels=symbols, fontsize=14)
plt.colorbar(im)
plt.tight_layout()
plt.show()
