#!/bin/bash

tmp_lang=data/local
input_lang=data/local/input/
output_lang=data/local/lang/
final_lang=data/lang
model_dir=kaldi_model
silence_phone=SIL
noise_phone=SPN
noise_word="<unk>"

mkdir -p $input_lang
echo SIL > $input_lang/silence_phones.txt
echo SPN >> $input_lang/silence_phones.txt
echo SIL > $input_lang/optional_silence.txt
cut -d ' ' -f 2- $model_dir/lexicon.txt | sed 's/ /\n/g' \
																				| awk "{ if(\$1 != \"$silence_phone\") print \$1; }" \
																				| awk "{ if(\$1 != \"$noise_phone\") print \$1; }" \
																				| sort -u > $input_lang/nonsilence_phones.txt
cp $model_dir/words.txt $input_lang
cp $model_dir/phones.txt $input_lang
cp $model_dir/lexicon.txt $input_lang
cp $model_dir/lexiconp.txt $input_lang
cp $model_dir/lexiconp_disambig.txt $input_lang
echo utils/prepare_lang.sh --phone-symbol-table $input_lang/phones.txt \
											$input_lang "$noise_word" $tmp_lang $output_lang
utils/prepare_lang.sh --phone-symbol-table $input_lang/phones.txt \
											$input_lang "$noise_word" $tmp_lang $output_lang

mkdir -p $final_lang/phones
cp $model_dir/disambig.int $final_lang/phones
cp $model_dir/words.txt $final_lang/
cp $model_dir/phones.txt $final_lang/
cp $model_dir/L_disambig.fst $final_lang/L.fst
cp $model_dir/L_disambig.fst $final_lang/
cp $model_dir/G.fst $final_lang/
echo "<unk>" > $final_lang/oov.txt
echo 18 > $final_lang/oov.int

#echo "!OOV OOV" > data/local/lang/lexicon.txt
#cat kaldi_model/lexicon.txt >> data/local/lang/lexicon.txt
#
#echo "!OOV 1.0 OOV" > data/local/lang/lexiconp.txt
#cat kaldi_model/lexiconp.txt >> data/local/lang/lexiconp.txt
#
#
#utils/prepare_lang.sh data/local/lang '!OOV' data/local/ data/lang
