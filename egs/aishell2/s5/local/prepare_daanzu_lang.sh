#!/bin/bash

stage=0
use_gpu=false

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
. ../../../tools/env.sh

tmp_lang=data/local
input_lang=data/input/
output_lang=data/lang/
model_dir=kaldi_model
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
  cp $model_dir/nonterminals.txt $input_lang
  sort $model_dir/lexicon.txt $model_dir/user_lexicon.txt > $input_lang/lexicon.txt
  cat $model_dir/user_lexicon.txt | sed "s/\(^[A-Za-z0-9-]*\>\)/\1 1.0/g" \
                                  | sort $model_dir/lexiconp.txt - > $input_lang/lexiconp.txt

  sort $model_dir/lexiconp_disambig.txt > $input_lang/lexiconp_disambig.txt

  # Generate language model aka L.fst/L_disambig.fst
  utils/prepare_lang.sh --phone-symbol-table $input_lang/phones.txt \
                        $input_lang "$noise_word" $tmp_lang $output_lang
fi

# Create a corresponding grammar and decoding graph
if [ $stage -le 1 ]; then
  # Generate a simple grammar aka G.fst
  ngram-count -order $lm_order -write-vocab $tmp_lang/vocab-full.txt \
              -wbdiscount -text data/train/corpus.txt -lm $tmp_lang/lm.arpa
  arpa2fst --disambig-symbol=\#0 --read-symbol-table=$output_lang/words.txt \
           $tmp_lang/lm.arpa $output_lang/G.fst

  # Alternatively use grammar shipped with daanzu model (incompatible with user lexicon)
  # cp $model_dir/G.fst $output_lang

  # Generate decoding graph aka HCLG.fst
  utils/mkgraph.sh --self-loop-scale 1.0 $output_lang $model_dir $graph_own_dir || exit 1;
fi

# Check decoding graph against test data
  cp $model_dir/final.mdl data/
	test_ivec_opt="--online-ivector-dir exp/nnet3_chain/ivectors_test_hires"
  steps/nnet3/decode.sh --use-gpu $use_gpu --acwt 1.0 --post-decode-acwt 10.0 \
    --scoring-opts "--min-lmwt 1" \
    --nj 8 --cmd "$decode_cmd" $test_ivec_opt \
    $graph_own_dir data/test_hires data/decode || exit 2;
