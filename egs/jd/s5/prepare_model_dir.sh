#!/bin/bash

stage=0
use_gpu=true
dir=exp/chain_cleaned/tdnn_1d_sp
ivector_extractor=exp/nnet3_cleaned/extractor
nj=10
minimal=true

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
. ../../../tools/env.sh

if [ $minimal == "false" ]; then
	cp -r kaldi_model final_model
else
	mkdir final_model
	mkdir final_model/conf
	mkdir final_model/ivector_extractor
fi

cp conf/mfcc.conf final_model/conf
cp conf/mfcc_hires.conf final_model/conf
cp conf/online_cmvn.conf final_model/conf
cp exp/nnet3_cleaned/extractor/splice_opts final_model/conf/splice.conf
cp exp/nnet3_cleaned/ivectors_jd_ls_100_clean_sp_hires/conf/ivector_extractor.conf final_model/conf

cp exp/nnet3_cleaned/extractor/final.* final_model/ivector_extractor
cp exp/nnet3_cleaned/extractor/global_cmvn.stats final_model/ivector_extractor

cp exp/chain_cleaned/tdnn_1d_sp/final.mdl final_model/
cp exp/chain_cleaned/tdnn_1d_sp/tree final_model/

tar -czf final.tar.gz final_model/ 
rm -r final_model
