#!/usr/bin/env bash

stage=1
nj=20

train_set=train
test_set=test
use_gpu=false

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

misrecognitions=misrecognitions.log
model_dir=kaldi_model
graph_own_dir=$model_dir/graph_own
data_set=${train_set}
data_dir=data/${data_set}

# Renice this script to low priority to minimize interference with other work
renice -n 19 $$

# Checks if g2p_en (required for determining phones of OOVs) is installed
python3 <<EOF
import sys
try:
	import g2p_en
except ImportError:
	sys.exit(1)    

sys.exit(0)
EOF
(($?)) && echo "g2p_en python3 library not available. please install." && exit 1

if [ $stage -le 1 ]; then
	local/prepare_data.py
fi

if [ $stage -le 2 ]; then
	[ ! -d data/local ] && mkdir -p data/local
	[ -e data/local/corpus.txt ] && rm data/local/corpus.txt

	echo "Generating corpus"
  for datadir in train `find data/ -type d -name "test*" -printf "%f\n"` ; do
		if [ ! -f data/$datadir/corpus.txt ]; then
			echo "corpus.txt not found in $datadir. generating"
			cat data/$datadir/text | awk '{$1=""; print $0}' | sed 's/^ *//' > data/$datadir/corpus.txt
		fi
		cat data/$datadir/corpus.txt >> data/local/corpus.txt
  done

	echo "Preparing language model"
	# Prepare language model compatible with KAG
	local/prepare_daanzu_lang.sh --model_dir kaldi_model/ \
															 --output_lang data/lang_nosp \
															 --corpus data/local
fi

if [ $stage -le 3 ]; then
	for test_set in data/test_* ; do
		if [[ "$test_set" =~ "_hires" ]]; then
			continue
		fi
		test_set=${test_set:5}

		echo "$0: creating high-resolution MFCC features for directory: ${test_set}"
		mfccdir=data/${test_set}_hires/data

		utils/copy_data_dir.sh data/$test_set data/${test_set}_hires

		steps/make_mfcc.sh --nj ${nj} --mfcc-config conf/mfcc_hires.conf \
			--cmd $train_cmd data/${test_set}_hires || exit 1;
		steps/compute_cmvn_stats.sh data/${test_set}_hires
		utils/fix_data_dir.sh data/${test_set}_hires

		test_set=${test_set}_hires
		steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj ${nj} \
			data/${test_set} $model_dir/ivector_extractor exp/nnet3_chain/ivectors_${test_set} || exit 1;
	done
fi

if [ $stage -le 5 ]; then
	for test_set in data/test_*; do
		if [[ "$test_set" =~ "_hires" ]]; then
			continue
		fi
		test_set=${test_set:5}

		# Score test data with reference model
		cp $model_dir/final.mdl data/
		test_ivec_opt="--online-ivector-dir exp/nnet3_chain/ivectors_${test_set}_hires"
		steps/nnet3/decode.sh --use-gpu $use_gpu --acwt 1.0 --post-decode-acwt 10.0 \
			--scoring-opts "--min-lmwt 1" \
			--nj ${nj} --num-threads ${nj} --cmd "$decode_cmd" $test_ivec_opt \
			$graph_own_dir data/${test_set}_hires data/decode_${test_set:5} || exit 2;
	done
fi

if [ $stage -le 6 ]; then
  > $misrecognitions

	for decode_set in data/decode_*; do
		if [[ "$decode_set" =~ "_hires" ]]; then
			continue
		fi
    cat ${decode_set}/scoring_kaldi/wer_details/per_utt \
      | grep -e " op *I\|S\|D" -B 2 \
      | cut -b 9- \
      | sed "s/\(.*_[0-9]\{6\}\)-\([a-z0-9-]*\)/\2 \1/g" \
      >> $misrecognitions

	done
fi


