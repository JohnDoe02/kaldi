#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from random import randrange

if os.path.exists("./data"):
    print("ERROR: data directory already exists. Aborting")
    exit(1)

def createRecordingIds(recordings, directory):
    recording_ids = []

    for name, postfix in zip(recordings, directory):
        name_begin = len(name) - name[::-1].find("/")
        name = name[name_begin:]
        name = name.replace("recorder_", "")
        name = name.replace("retain_", "")
        name = name.replace(".wav", "")
        prefix = "{0:07d}".format(randrange(0, 9999999))
        recording_ids.append(prefix + "-" + name + "-" + postfix)

    return recording_ids

def writeTextFile(text_file, ids, recognitions):
    f = open(text_file, "w")
    for id, recognition in zip(ids, recognitions):
        f.write(id + " " + recognition + "\n")
    f.close()

def writeSpeakerToGender(stg_file, speakers, gender):
    f = open(stg_file, "w")
    for speaker in speakers:
        f.write(speaker + " " + gender + "\n")
    f.close()

def writeSpeakerToUtterance(stu_file, speakers, ids):
    f = open(stu_file, "w")
    for id, speaker in zip(ids, speakers):
        f.write(speaker + " " + id + "\n")
    f.close()

train_file = ("dataset.tsv", "train")
test_files = [(filename, filename.replace(".tsv", ""))
                for filename in os.listdir('dataset/') if filename.startswith("test_")]

for filename, directory in [train_file] + test_files:
    speaker = "speaker"
    gender = "m"
    recordings = pd.read_table("dataset/" + filename, header=0, 
                               names=["File", "Length", "Directory", "Recognition"])

    recordings["IDs"] = createRecordingIds(recordings["File"], recordings["Directory"])
    #It's essential here to prepend speaker after ID to avoid sorting issues with some kaldi tools
    recordings["Speaker"] = recordings["IDs"] + "-" + speaker

    dataset = recordings.sort_values(by="IDs")

    print("Writing data directory:", directory)
    os.makedirs("data/" + directory)
    print("")
    scp_file = "data/" + directory + "/wav.scp"
    print("Writing scp file: {} .. ".format(scp_file), end='')
    dataset.to_csv(scp_file, sep=" ", header=0, index=False, columns=["IDs","File"])
    print("done.")

    uts_file = "data/" + directory + "/utt2spk"
    print("Writing utt2spk file: {} .. ".format(uts_file), end='')
    dataset.to_csv(uts_file, sep=" ", header=0, index=False, columns=["IDs","Speaker"])
    print("done.")

    stu_file = "data/" + directory + "/spk2utt"
    print("Writing spk2utt file: {} .. ".format(stu_file), end='')
    writeSpeakerToUtterance(stu_file, dataset["Speaker"], dataset["IDs"])
    print("done.")

    corpus_file = "data/" + directory + "/corpus.txt"
    print("Writing corpus file: {} .. ".format(corpus_file), end='')
    dataset.to_csv(corpus_file, sep="\t", header=0, index=False, columns=["Recognition"])
    print("done.")

    text_file = "data/" + directory + "/text"
    print("Writing text file: {} .. ".format(text_file), end='')
    writeTextFile(text_file, dataset["IDs"], dataset["Recognition"])
    print("done.")

    stg_file = "data/" + directory + "/spk2gender"
    print("Writing spk2gender file: {} .. ".format(stg_file), end='')
    writeSpeakerToGender(stg_file, dataset["Speaker"], gender)
    print("done.")
    print("")
    print("")
