#!/bin/bash

mkdir -p data/lang/phones
cp kaldi_model/disambig.int data/lang/phones
cp kaldi_model/words.txt data/lang/
cp kaldi_model/phones.txt data/lang/
cp kaldi_model/L_disambig.fst data/lang/L.fst
#utils/make_lexicon_fst.pl kaldi_model/lexicon.txt 0.5 SIL | \
#  fstcompile --isymbols=kaldi_model/phones.txt --osymbols=kaldi_model/words.txt \
#  --keep_isymbols=false --keep_osymbols=false | \
#   fstarcsort --sort_type=olabel > data/lang/L.fst
echo 18 > data/lang/oov.int

#echo "!OOV OOV" > data/local/lang/lexicon.txt
#cat kaldi_model/lexicon.txt >> data/local/lang/lexicon.txt
#
#echo "!OOV 1.0 OOV" > data/local/lang/lexiconp.txt
#cat kaldi_model/lexiconp.txt >> data/local/lang/lexiconp.txt
#
#cut -d ' ' -f 2- kaldi_model/lexicon.txt | sed 's/ /\n/g' | awk '{ if($1 != "SIL") print $1; }' | sort -u > data/local/lang/nonsilence_phones.txt
#echo SIL > data/local/lang/silence_phones.txt
#echo OOV >> data/local/lang/silence_phones.txt
#echo SIL > data/local/lang/optional_silence.txt
#
#utils/prepare_lang.sh data/local/lang '!OOV' data/local/ data/lang
