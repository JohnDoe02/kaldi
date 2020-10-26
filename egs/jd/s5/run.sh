#!/usr/bin/env bash

librispeech_datasets=(dev-clean test-clean dev-other test-other train-clean-100)
librispeech_url=www.openslr.org/resources/12
librispeech=/mnt/copora/librispeech/
stage=1
nj=20

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

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
  # download the data.  Note: we're using the 100 hour setup for
  # now; later in the script we'll download more and use it to train neural
  # nets.
  for part in "${librispeech_datasets[@]}"; do
    local/download_and_untar.sh $librispeech $librispeech_url $part
  done

  # ... and then combine the two sets into a 460 hour one
  #utils/combine_data.sh \
  #  data/train_clean_460 data/train_clean_100 data/train_clean_360

  # download the LM resources
  #local/download_lm.sh $lm_url data/local/lm
fi

if [ $stage -le 3 ]; then
  # format the data as Kaldi data directories
  for part in "${librispeech_datasets[@]}"; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $librispeech/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done
fi

# you might not want to do this for interactive shells.
#set -e

if [ $stage -le 5 ]; then
  for part in train_command train_dictation train_day-to-day "${librispeech_datasets[@]//-/_}"; do

    steps/make_mfcc.sh --cmd "$train_cmd" --nj ${nj} data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done
fi

if [ $stage -le 6 ]; then
  # Make some small data subsets for early system-build stages.  Note, there are 29k
  # utterances in the train_clean_100 directory which has 100 hours of data.
  # For the monophone stages we select the shortest utterances, which should make it
  # easier to align the data from a flat start.

  utils/subset_data_dir.sh --shortest data/train_command 500 data/jd_command_500short
  utils/subset_data_dir.sh --shortest data/train_dictation 500 data/jd_dictation_500short
  utils/subset_data_dir.sh --shortest data/train_clean_100 1000 data/ls_1kshort
	utils/combine_data.sh data/train_2kshort \
		data/jd_command_500short data/jd_dictation_500short data/ls_1kshort

  utils/subset_data_dir.sh data/train_command 1250 data/jd_command_1250
  utils/subset_data_dir.sh data/train_dictation 1250 data/jd_dictation_1250
  utils/subset_data_dir.sh data/train_clean_100 2500 data/ls_2.5k
	utils/combine_data.sh data/train_5k data/jd_command_1250 data/jd_dictation_1250 data/ls_2.5k

  utils/subset_data_dir.sh data/train_command 2500 data/jd_command_2500
  utils/subset_data_dir.sh data/train_dictation 2500 data/jd_dictation_2500
  utils/subset_data_dir.sh data/train_clean_100 5000 data/ls_5k
	utils/combine_data.sh data/train_10k data/jd_command_2500 data/jd_dictation_2500 data/ls_5k

	utils/combine_data.sh data/jd_ls_100_clean data/train_command data/train_dictation data/train_day-to-day data/train_clean_100
fi

if [ $stage -le 7 ]; then
	[ ! -d data/local ] && mkdir -p data/local
	[ -e data/local/corpus.txt ] && rm data/local/corpus.txt

	echo "Generating corpus"
  for datadir in train test_day-to-day test_command test_dictation "${librispeech_datasets[@]//-/_}"; do
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

	# This copy is not really necessary
	# only for simplifying compatibility with librispeech recipe
	cp -r data/lang_nosp data/lang

  local/format_lms.sh --src-dir data/lang_nosp data/local/lm
fi

if [ $stage -le 8 ]; then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj ${nj} --cmd "$train_cmd" \
                      data/train_2kshort data/lang_nosp exp/mono
fi

if false && [ $stage -le 9 ]; then
	utils/mkgraph.sh --mono data/lang_nosp exp/mono exp/mono/graph || exit 1
	steps/decode.sh --config conf/decode.config --nj ${nj} --cmd "$decode_cmd" exp/mono/graph data/test_command exp/mono/decode
	#steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/mono/graph data/test_dictation exp/mono/decode
fi


if [ $stage -le 9 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj ${nj} --cmd "$train_cmd" \
                    data/train_5k data/lang_nosp exp/mono exp/mono_ali_5k

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2000 10000 data/train_5k data/lang_nosp exp/mono_ali_5k exp/tri1
fi

if [ $stage -le 10 ]; then
  steps/align_si.sh --nj ${nj} --cmd "$train_cmd" \
                    data/train_10k data/lang_nosp exp/tri1 exp/tri1_ali


  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                          data/train_10k data/lang_nosp exp/tri1_ali exp/tri2b
fi

if [ $stage -le 11 ]; then
  # Align a 10k utts subset using the tri2b model
  steps/align_si.sh  --nj ${nj} --cmd "$train_cmd" --use-graphs true \
                     data/train_10k data/lang_nosp exp/tri2b exp/tri2b_ali

  # Train tri3b, which is LDA+MLLT+SAT on 10k utts
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
                     data/train_10k data/lang_nosp exp/tri2b_ali exp/tri3b

fi

if [ $stage -le 12 ]; then
  # align the entire train_clean_100 subset using the tri3b model
  steps/align_fmllr.sh --nj ${nj} --cmd "$train_cmd" \
    data/jd_ls_100_clean data/lang_nosp \
    exp/tri3b exp/tri3b_ali_clean_100

  # train another LDA+MLLT+SAT system on the entire 100 hour subset
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
                      data/jd_ls_100_clean data/lang_nosp \
                      exp/tri3b_ali_clean_100 exp/tri4b
fi

#if [ $stage -le 13 ]; then
#  # align the entire train_clean_100 subset using the tri3b model
#  steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
#    data/jd_ls_100_clean data/lang_nosp \
#    exp/tri4b exp/tri4b_ali_clean_100
#fi


if [ $stage -le 20 ]; then
  # train and test nnet3 tdnn models on the entire data with data-cleaning.
  local/chain/run_tdnn.sh --train_set jd_ls_100_clean \
													--gmm tri4b \
													--stage 15 # set "--stage 11" if you have already run local/nnet3/run_tdnn.sh
fi

exit 0


if [ $stage -le 13 ]; then
  # Now we compute the pronunciation and silence probabilities from training data,
  # and re-create the lang directory.
  steps/get_prons.sh --cmd "$train_cmd" \
                     data/train_clean_100 data/lang_nosp exp/tri4b
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
                                  data/local/dict_nosp \
                                  exp/tri4b/pron_counts_nowb.txt exp/tri4b/sil_counts_nowb.txt \
                                  exp/tri4b/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict \
                        "<UNK>" data/local/lang_tmp data/lang
  local/format_lms.sh --src-dir data/lang data/local/lm

  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge
fi

if [ $stage -le 14 ] && false; then
  # This stage is for nnet2 training on 100 hours; we're commenting it out
  # as it's deprecated.
  # align train_clean_100 using the tri4b model
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train_clean_100 data/lang exp/tri4b exp/tri4b_ali_clean_100

  # This nnet2 training script is deprecated.
  local/nnet2/run_5a_clean_100.sh
fi

if [ $stage -le 15 ]; then
  local/download_and_untar.sh $data $data_url train-clean-360

  # now add the "clean-360" subset to the mix ...
  local/data_prep.sh \
    $data/LibriSpeech/train-clean-360 data/train_clean_360
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/train_clean_360 \
                     exp/make_mfcc/train_clean_360 $mfccdir
  steps/compute_cmvn_stats.sh \
    data/train_clean_360 exp/make_mfcc/train_clean_360 $mfccdir

  # ... and then combine the two sets into a 460 hour one
  utils/combine_data.sh \
    data/train_clean_460 data/train_clean_100 data/train_clean_360
fi

if [ $stage -le 16 ]; then
  # align the new, combined set, using the tri4b model
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
                       data/train_clean_460 data/lang exp/tri4b exp/tri4b_ali_clean_460

  # create a larger SAT model, trained on the 460 hours of data.
  steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
                      data/train_clean_460 data/lang exp/tri4b_ali_clean_460 exp/tri5b
fi


# The following command trains an nnet3 model on the 460 hour setup.  This
# is deprecated now.
## train a NN model on the 460 hour set
#local/nnet2/run_6a_clean_460.sh

if [ $stage -le 17 ]; then
  # prepare the remaining 500 hours of data
  local/download_and_untar.sh $data $data_url train-other-500

  # prepare the 500 hour subset.
  local/data_prep.sh \
    $data/LibriSpeech/train-other-500 data/train_other_500
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/train_other_500 \
                     exp/make_mfcc/train_other_500 $mfccdir
  steps/compute_cmvn_stats.sh \
    data/train_other_500 exp/make_mfcc/train_other_500 $mfccdir

  # combine all the data
  utils/combine_data.sh \
    data/train_960 data/train_clean_460 data/train_other_500
fi

if [ $stage -le 18 ]; then
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
                       data/train_960 data/lang exp/tri5b exp/tri5b_ali_960

  # train a SAT model on the 960 hour mixed data.  Use the train_quick.sh script
  # as it is faster.
  steps/train_quick.sh --cmd "$train_cmd" \
                       7000 150000 data/train_960 data/lang exp/tri5b_ali_960 exp/tri6b

  # decode using the tri6b model
  utils/mkgraph.sh data/lang_test_tgsmall \
                   exp/tri6b exp/tri6b/graph_tgsmall
  for test in test_clean test_other dev_clean dev_other; do
      steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
                            exp/tri6b/graph_tgsmall data/$test exp/tri6b/decode_tgsmall_$test
      steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
                         data/$test exp/tri6b/decode_{tgsmall,tgmed}_$test
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
        data/$test exp/tri6b/decode_{tgsmall,tglarge}_$test
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
        data/$test exp/tri6b/decode_{tgsmall,fglarge}_$test
  done
fi


if [ $stage -le 19 ]; then
  # this does some data-cleaning. The cleaned data should be useful when we add
  # the neural net and chain systems.  (although actually it was pretty clean already.)
  local/run_cleanup_segmentation.sh
fi

# steps/cleanup/debug_lexicon.sh --remove-stress true  --nj 200 --cmd "$train_cmd" data/train_clean_100 \
#    data/lang exp/tri6b data/local/dict/lexicon.txt exp/debug_lexicon_100h

# #Perform rescoring of tri6b be means of faster-rnnlm
# #Attention: with default settings requires 4 GB of memory per rescoring job, so commenting this out by default
# wait && local/run_rnnlm.sh \
#     --rnnlm-ver "faster-rnnlm" \
#     --rnnlm-options "-hidden 150 -direct 1000 -direct-order 5" \
#     --rnnlm-tag "h150-me5-1000" $data data/local/lm

# #Perform rescoring of tri6b be means of faster-rnnlm using Noise contrastive estimation
# #Note, that could be extremely slow without CUDA
# #We use smaller direct layer size so that it could be stored in GPU memory (~2Gb)
# #Suprisingly, bottleneck here is validation rather then learning
# #Therefore you can use smaller validation dataset to speed up training
# wait && local/run_rnnlm.sh \
#     --rnnlm-ver "faster-rnnlm" \
#     --rnnlm-options "-hidden 150 -direct 400 -direct-order 3 --nce 20" \
#     --rnnlm-tag "h150-me3-400-nce20" $data data/local/lm


if [ $stage -le 20 ]; then
  # train and test nnet3 tdnn models on the entire data with data-cleaning.
  local/chain/run_tdnn.sh # set "--stage 11" if you have already run local/nnet3/run_tdnn.sh
fi

# The nnet3 TDNN recipe:
# local/nnet3/run_tdnn.sh # set "--stage 11" if you have already run local/chain/run_tdnn.sh

# # train models on cleaned-up data
# # we've found that this isn't helpful-- see the comments in local/run_data_cleaning.sh
# local/run_data_cleaning.sh

# # The following is the current online-nnet2 recipe, with "multi-splice".
# local/online/run_nnet2_ms.sh

# # The following is the discriminative-training continuation of the above.
# local/online/run_nnet2_ms_disc.sh

# ## The following is an older version of the online-nnet2 recipe, without "multi-splice".  It's faster
# ## to train but slightly worse.
# # local/online/run_nnet2.sh
