#!/usr/bin/env bash

. ./path.sh

# This script needs:
# [in data/local/dict/ ]
# lexicon_common_voice_uk.txt
#
# The parts of the output of this that will be needed later are
# [in data/local/dict/ ]
# lexicon.txt
# extra_questions.txt
# nonsilence_phones.txt
# optional_silence.txt
# silence_phones.txt

dir=data/local/dict

test -f $dir/lexicon_common_voice_uk.txt

#for w in sil laughter noise oov; do echo $w; done > $dir/silence_phones.txt
for w in sil; do echo $w; done > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

# An extra question will be added by including the silence phones in one class.
cat $dir/silence_phones.txt|  awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;

conflate_phones() {
    #sed 's,ZZZZ,ZZ,g'
    cat
}

# Phone alphabet
cut -s -f2- $dir/lexicon_common_voice_uk.txt | tr ' ' '\n' | conflate_phones | sort -u > $dir/nonsilence_phones.txt

# Base lexicon
cut -s -f1- $dir/lexicon_common_voice_uk.txt | sort -u > $dir/lexicon1_raw_nosil.txt

# Add prons for silence_phones
for w in `grep -v sil $dir/silence_phones.txt`; do
  echo "[$w] $w"
done | cat - $dir/lexicon1_raw_nosil.txt  > $dir/lexicon2_raw.txt || exit 1;

# Cat your local lexicon file if you have it
cat $dir/lexicon2_raw.txt > $dir/lexicon3_expand.txt

cat $dir/lexicon3_expand.txt  <( echo "<unk> sil" ) > $dir/lexicon4_extra.txt

conflate_phones < $dir/lexicon4_extra.txt > $dir/lexicon.txt

utils/validate_dict_dir.pl $dir

# ref https://github.com/kaldi-asr/kaldi/blob/master/egs/tedlium/s5_r2/local/run_unk_model.sh
utils/lang/make_unk_lm.sh "$dir" exp/make_unk
