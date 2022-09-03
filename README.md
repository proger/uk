# Вирівняти довгі аудіо та тексти українською

This recipe trains a Ukrainian GMM-HMM on Common Voice 9.0 to use
for segmentation of long audio files into short utterances using its full transcript.

It avoids performing full large vocabulary speech recognition
by limiting its search options to word sequences from the input transcript.

## Recipe

### Install Kaldi

```bash
git clone https://github.com/kaldi-asr/kaldi
cd kaldi
cat INSTALL
# install kaldi to $HOME/kaldi
```

Speed run through Kaldi installation instructions:

```
cd tools
./extras/check_dependencies.sh
# Pay attention to dependency errors.
# If you're on macOS you don't need OpenBLAS.
# You won't need python2.7 and subversion.
# Make sure python command runs some python:
ln -sf $(which python3) $HOME/.local/bin/python

# Ignore all subsequent dependency checks.
cat > extras/check_dependencies.sh

# Build all tools (primarily openfst and pocolm)
make -j8
./extras/install_pocolm.sh

# Build kaldi itself
cd ../src
./configure --shared
make -j clean depend
make -j8
```

### Install uk

```
pip3 install --editable .
```

### Train Ukrainian using Common Voice 10.0

Prerequisites:

- Request access at https://huggingface.co/datasets/mozilla-foundation/common_voice_10_0
- Get a Hugging Face token at https://huggingface.co/settings/tokens
- run `huggingface-cli login`

```bash
# bring parts of kaldi into $PATH
source path.sh

# prepare dataset for training
# FWIW downloading takes longer than training :)
python3 -m uk.prepare_dataset

# progressively train mono, tri, tri2b, tri3b models
python3 -m uk.train_gmm
```

### Train English using LibriSpeech train-clean-100

```bash
python3 -m uk.prepare_dataset --dataset darkproger/librispeech_asr --subset train.clean.100 --split full --alphabet latin

# bring parts of kaldi into $PATH
source path.sh

# make a subset of librispeech
utils/subset_data_dir.sh --per-spk data/darkproger/librispeech_asr/train.clean.100/full 30 data/librispeech_mini

# progressively train mono, tri, tri2b, tri3b models
python3 -m uk.train_gmm -d data/librispeech_mini --dictdir data/english exp/english
```

### Segment Parallel Speech and Text Data

```bash
# get example data to align
git clone https://github.com/lang-uk/semesyuk-to-text

# note: text can contain extra words
cat semesyuk-to-text/texts/tokenized/semesyuk_farshrutka/01_prologue.txt | python3 -m uk.nlp_uk_tokens \
    > data/local/semesyuk_farshrutka/01_prologue.txt

# run segmentation using tri3b model
# in this example it outputs a kaldi-style data directory to data/semesyuk_farshrutka_prologue
python3 -m uk.segment_long_utterances -w exp/segment1 -o data/semesyuk_farshrutka_prologue \
    data/local/semesyuk_farshrutka/01_prologue.txt \
    semesyuk-to-text/audio/raw/semesyuk_farshrutka/01_prologue.mp3

# upload result to wandb
python3 -m uk.share data/semesyuk_farshrutka_prologue
```

## Acknowledgements

- Common Voice Dataset https://commonvoice.mozilla.org/
- G2P model https://github.com/kosti4ka/ukro_g2p
- Baseline https://github.com/lang-uk/semesyuk-to-text
- Alignment method (paper inside) https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/cleanup/segment_long_utterances.sh
