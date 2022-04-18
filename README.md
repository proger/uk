# Вирівняти Семесюка

This recipe trains a Ukrainian GMM-HMM on Common Voice 8.0 to use
for segmentation of long audio files into short utterances using its full transcript.

It avoids performing full large vocabulary speech recognition
by limiting its search options to word sequences from the input transcript.

## Recipe

### Kaldi

```bash
git clone https://github.com/kaldi-asr/kaldi
# install kaldi to $HOME/kaldi
```

### This Library

Prerequisites:

- Install ffmpeg
- Share contact information at https://huggingface.co/datasets/mozilla-foundation/common_voice_8_0
- Get a Hugging Face token at https://huggingface.co/settings/tokens

```bash
pip3 install --editable .

export HF_AUTH_TOKEN=hf_yolo

. path.sh

# prepare dataset for training
python3 -m uk.common_voice

# progressively train mono, tri, tri2b, tri3b models
python3 -m uk.train_gmm

# get data to align
git clone https://github.com/lang-uk/semesyuk-to-text

cat semesyuk-to-text/texts/tokenized/semesyuk_farshrutka/01_prologue.txt | python3 -m uk.nlp_uk_tokens \
    > data/local/semesyuk_farshrutka/01_prologue.txt

# run alignment using tri3b model
python3 -m uk.segment_long_utterances -w exp/segment1 -o data/semesyuk_farshrutka_prologue \
    data/local/semesyuk_farshrutka/01_prologue.txt \
    semesyuk-to-text/audio/raw/semesyuk_farshrutka/01_prologue.mp3
```

## Acknowledgements

- Common Voice Dataset https://commonvoice.mozilla.org/
- G2P model https://github.com/kosti4ka/ukro_g2p
- Baseline https://github.com/lang-uk/semesyuk-to-text
