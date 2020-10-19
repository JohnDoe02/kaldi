#!/usr/bin/env python3

from datetime import datetime
import subprocess
import os
import shutil

default_params = {
        "num_epochs":1,
        "initial_effective_lrate":.00075,
        "final_effective_lrate":.000075,

        "tree_dir":"tree_sp/",
        "stage":10,
        "nj":10,

        "minibatch_size": [128,64,32,16],
        "xent_regularize":0.1,
        "train_stage":-2,
        "get_egs_stage":-10,
        "dropout_schedule":'0,0@0.20,0.5@0.50,0',
        "frames_per_eg":[150,110,100,40],
        "phone_lm_scales":(1,100),
        "primary_lr_factor":0.25,
}

def write_summary(dest, exp):
    with open(dest, 'w') as outfile:
        for fname in exp:
            with open(fname + "/scoring_kaldi/best_wer" ) as infile:
                outfile.write(infile.read())

def get_testsets(datadir):
    if(datadir[-1] == "/"):
        datadir = datadir[:-1]

    return [datadir + "/" + x for x in os.listdir(datadir) if x.startswith("decode_")]

call = ["./local/chain/tuning/run_finetune_tdnn_1a_daanzu.sh"]
for flag,value in default_params.items():
    call.append("--{}".format(flag))
    call.append(str(value).strip("[]()").replace(" ",""))


print(" ".join(call))
#subprocess.run(call)

print("Training complete, saving results")
now = datetime.now()
now_string = now.strftime("%d-%m-%Y_%H-%M-%S")
target = "results/" + now_string
print(now_string)
#ref = get_testsets("data/")
exp = get_testsets("exp/nnet3_chain/train")
os.makedirs(target)
write_summary(target + "/summary.txt", exp)
shutil.move("exp/nnet3_chain/train", target)
