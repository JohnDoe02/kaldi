#!/usr/bin/env python3

from datetime import datetime
import subprocess
import os
import shutil
import sys
import numpy as np

default_params = {
        "num_epochs":1,
        "initial_effective_lrate":.00025,
        "final_effective_lrate":.000025,

        "tree_dir":"tree_sp/",
        "stage":9,
        "nj":10,

        "minibatch_size": [128,64,32,16],
        "xent_regularize":0.1,
        "train_stage":-4,
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

def get_call(params):
    call = ["./local/chain/tuning/run_finetune_tdnn_1a_daanzu.sh"]
    for flag,value in params.items():
        call.append("--{}".format(flag))
        call.append(str(value).strip("[]()").replace(" ",""))

    return call

def param_to_txt(param):
    ret = ""
    for key, value in params.items():
        if isinstance(value, str):
            value = "\"" + value + "\""

        ret += key + ": " + str(value) + ",\n"

    return ret

def call_and_log(call, logfile):
    process = subprocess.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open(logfile, 'ab') as file:
        for line in process.stdout:
            file.write(line)
            sys.stdout.buffer.write(line)
            sys.stdout.flush()

    process.wait()

#initial_effective_lrates = np.arange(.0001, .0005, .0002)
#final_effective_lrates = np.arange(.00001, .00005, .00002)
phone_lm_scaless = [(0,1), (1,50)]

logfile = "training.log"
flagfile = "flags.txt"
summaryfile = "summary.txt"

test_stage = 1
for phone_lm_scales in phone_lm_scaless:
#    for initial_effective_lrate, final_effective_lrate\
#            in zip(initial_effective_lrates, final_effective_lrates):
    params = default_params
#        params["initial_effective_lrate"] = initial_effective_lrate
#        params["final_effective_lrate"] = final_effective_lrate
    params["phone_lm_scales"] = phone_lm_scales

    print("Starting training cycle with parameters:")
    print(param_to_txt(params))
    print("")
    print("")

    with open(flagfile, 'w') as file:
        file.write(param_to_txt(params))

    call = get_call(params)
    call_and_log(call, logfile)

    call = ["./local/run_tests.sh", "--stage", str(test_stage)]
    test_stage = 1

    call_and_log(call, logfile)

    print("Training complete, saving results")
    now = datetime.now()
    now_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    target = "results/" + now_string
    print(now_string)
    #ref = get_testsets("data/")
    exp = get_testsets("exp/nnet3_chain/train")
    os.makedirs(target)
    write_summary(target + "/" + summaryfile, exp)
    shutil.copytree("exp/nnet3_chain/train", target + "/train")
    shutil.move(logfile, target)
    shutil.move(flagfile, target)
