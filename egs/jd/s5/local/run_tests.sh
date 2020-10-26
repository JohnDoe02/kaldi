#!/bin/bash

stage=0
use_gpu=true
dir=exp/chain_cleaned/tdnn_1d_sp
ivector_extractor=exp/nnet3_cleaned/extractor
nj=10

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
. ../../../tools/env.sh


if [ $stage -le 0 ]; then
	for test_set in train ; do
		if [[ "$test_set" =~ "_hires" ]]; then
			continue
		fi
		#test_set=${test_set:5}

		echo "$0: creating high-resolution MFCC features for directory: ${test_set}"
		mfccdir=data/${test_set}_hires/data

		utils/copy_data_dir.sh data/$test_set data/${test_set}_hires

		steps/make_mfcc.sh --nj ${nj} --mfcc-config conf/mfcc_hires.conf \
			--cmd $train_cmd data/${test_set}_hires || exit 1;
		steps/compute_cmvn_stats.sh data/${test_set}_hires
		utils/fix_data_dir.sh data/${test_set}_hires

		steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj ${nj} \
			data/${test_set}_hires $ivector_extractor $dir/ivectors_${test_set}_hires || exit 1;
	done
fi

if [ $stage -le 1 ]; then
	for test_set in train; do
		if [[ "$test_set" =~ "_hires" ]]; then
			continue
		fi
		#test_set=${test_set:5}

		# Note: it might appear that this $lang directory is mismatched, and it is as
		# far as the 'topo' is concerned, but this script doesn't read the 'topo' from
		# the lang directory.
		utils/mkgraph.sh --self-loop-scale 1.0 data/lang_nosp $dir $dir/graph
		steps/nnet3/decode.sh --use-gpu $use_gpu --acwt 1.0 --post-decode-acwt 10.0 \
			--scoring-opts "--min-lmwt 1" --frames_per_chunk 150 \
			--nj ${nj} --num-threads ${nj} --cmd "$decode_cmd" --online-ivector-dir $dir/ivectors_${test_set}_hires \
			$dir/graph data/${test_set}_hires $dir/decode_${test_set:5} || exit 1;
	done
fi
