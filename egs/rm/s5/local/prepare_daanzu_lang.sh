#!/bin/bash

stage=0
use_gpu=false

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

tmp_lang=data/local
input_lang=data/input/
output_lang=data/lang/
model_dir=kaldi_model
silence_phone=SIL
noise_phone=SPN
noise_word="<unk>"
graph_own_dir=$model_dir/graph_own

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
	cp $model_dir/lexicon.txt $input_lang
	cp $model_dir/lexiconp.txt $input_lang
	cp $model_dir/lexiconp_disambig.txt $input_lang
	utils/prepare_lang.sh --phone-symbol-table $input_lang/phones.txt \
												$input_lang "$noise_word" $tmp_lang $output_lang
fi

# Create a corresponding decoding graph
if [ $stage -le 1 ]; then

	# Generate simple G.fst
#	fstcompile --isymbols=$output_lang/words.txt --osymbols=$output_lang/words.txt \
#						 --keep_isymbols=false -keep_osymbols=false data/local/tmp/G.txt \
#		| fstarcsort --sort_type=ilabel > $output_lang/G.fst || exit 1;
	cp $model_dir/G.fst $output_lang

	utils/mkgraph.sh --self-loop-scale 1.0 $output_lang $model_dir $graph_own_dir || exit 1;
fi

# Check decoding graph against test data
if [ $stage -le -1 ]; then
	cp $model_dir/final.mdl data/
	test_ivec_opt="--online-ivector-dir exp/nnet2_online_wsj/ivectors_test"
	steps/nnet3/decode.sh --use-gpu $use_gpu --acwt 1.0 --post-decode-acwt 10.0 \
		--scoring-opts "--min-lmwt 1" \
		--nj 8 --cmd "$decode_cmd" $test_ivec_opt \
		$graph_own_dir data/test_hires data/decode || exit 2;
fi
