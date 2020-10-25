#!/bin/bash

stage=0
use_gpu=false

train=data/train
input_lang=data/input/
output_lang=data/lang/
model_dir=kaldi_model

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
. ../../../tools/env.sh

tmp_lang=data/local
silence_phone=SIL
noise_phone=SPN
noise_word="<unk>"
graph_own_dir=$model_dir/graph_own
lm_order=2

# Create a simple language model based on daanzu model
if [ $stage -le 0 ]; then
  mkdir -p $input_lang
  echo SIL > $input_lang/silence_phones.txt
  echo SPN >> $input_lang/silence_phones.txt
  echo NSN >> $input_lang/silence_phones.txt
  echo SIL > $input_lang/optional_silence.txt
  cut -d ' ' -f 2- $model_dir/lexicon.txt | sed 's/ /\n/g' \
                                          | awk "{ if(\$1 != \"$silence_phone\") print \$1; }" \
                                          | awk "{ if(\$1 != \"$noise_phone\") print \$1; }" \
                                          | sort -u > $input_lang/nonsilence_phones.txt
  cp $model_dir/phones.txt $input_lang
#  cp $model_dir/nonterminals.txt $input_lang
	if [ ! -f $train/corpus.txt ]; then
		echo "corpus.txt not found. generating"
		cat $train/text | awk '{$1=""; print $0}' | sed 's/^ *//' > $train/corpus.txt
	fi
	./local/generate_pronunciations.py --lexicon ${model_dir}/lexicon.txt \
																		 --corpus ${train}/corpus.txt \
																		 --lexicon_oov ${input_lang}/oov.txt \
																		 --phones ${input_lang}/phones.txt

	sort $model_dir/lexicon.txt $model_dir/user_lexicon.txt $input_lang/oov.txt \
		 > $input_lang/lexicon.txt

  cat $model_dir/user_lexicon.txt $input_lang/oov.txt | sed "s/\(^['A-Za-z0-9-]*\)/\1 1.0/g" \
																											| sort $model_dir/lexiconp.txt - \
																											> $input_lang/lexiconp.txt

  # Generate language model aka L.fst/L_disambig.fst
  utils/prepare_lang.sh --phone-symbol-table $input_lang/phones.txt \
                        $input_lang "$noise_word" $tmp_lang $output_lang
fi

# Create a corresponding grammar and decoding graph
if [ $stage -le 1 ]; then
  # Generate a simple grammar aka G.fst

  ngram-count -order $lm_order -write-vocab $tmp_lang/vocab-full.txt \
              -wbdiscount -text $train/corpus.txt -lm $tmp_lang/lm.arpa
  arpa2fst --disambig-symbol=\#0 --read-symbol-table=$output_lang/words.txt \
           $tmp_lang/lm.arpa $output_lang/G.fst

  # Alternatively use grammar shipped with daanzu model (incompatible with user lexicon)
  # cp $model_dir/G.fst $output_lang

  # Generate decoding graph aka HCLG.fst
  utils/mkgraph.sh --self-loop-scale 1.0 $output_lang $model_dir $graph_own_dir || exit 1;
fi
