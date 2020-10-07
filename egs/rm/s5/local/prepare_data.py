#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np

def createRecordingIds(recordings):
    recording_ids = []

    for name in recordings:
        name_begin = len(name) - name[::-1].find("/")
        name = name[name_begin:]
        name = name.replace("recording_", "")
        name = name.replace("retain_", "")
        name = name.replace(".wav", "")
        recording_ids.append(name)

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

speaker = "speaker"
gender = "m"
recordings = pd.read_table("dataset/dataset.tsv", header=0, 
                           names=["File", "Length", "Directory", "Recognition"])

recordings["IDs"] = createRecordingIds(recordings["File"])
recordings["Speaker"] = speaker + "-" + recordings["IDs"]

dataset = recordings

train = dataset.sample(frac=0.8, random_state=1337)
test = dataset.drop(train.index).sort_values(by="IDs")
train = dataset.drop(test.index).sort_values(by="IDs")

if os.path.exists("./data"):
    print("ERROR: data directory already exists. Aborting")
    exit(1)

print("Creating directories for train and test data .. ", end='')
os.makedirs("data/train")
os.makedirs("data/test")
print("done.")

print("")
print("")
print("Writing train data:")
print("")
scp_file = "data/train/wav.scp"
print("Writing scp file: {} .. ".format(scp_file), end='')
train.to_csv(scp_file, sep=" ", header=0, index=False, columns=["IDs","File"])
print("done.")

uts_file = "data/train/utt2spk"
print("Writing utt2spk file: {} .. ".format(uts_file), end='')
train.to_csv(uts_file, sep=" ", header=0, index=False, columns=["IDs","Speaker"])
print("done.")

stu_file = "data/train/spk2utt"
print("Writing spk2utt file: {} .. ".format(stu_file), end='')
writeSpeakerToUtterance(stu_file, train["Speaker"], train["IDs"])
print("done.")

corpus_file = "data/train/corpus.txt"
print("Writing corpus file: {} .. ".format(corpus_file), end='')
train.to_csv(corpus_file, sep="\t", header=0, index=False, columns=["Recognition"])
print("done.")

text_file = "data/train/text"
print("Writing text file: {} .. ".format(text_file), end='')
writeTextFile(text_file, train["IDs"], train["Recognition"])
print("done.")

stg_file = "data/train/spk2gender"
print("Writing spk2gender file: {} .. ".format(stg_file), end='')
writeSpeakerToGender(stg_file, train["Speaker"], gender)
print("done.")

print("")
print("")
print("Writing test data:")
print("")
scp_file = "data/test/wav.scp"
print("Writing scp file: {} .. ".format(scp_file), end='')
test.to_csv(scp_file, sep=" ", header=0, index=False, columns=["IDs","File"])
print("done.")

uts_file = "data/test/utt2spk"
print("Writing utt2spk file: {} .. ".format(uts_file), end='')
test.to_csv(uts_file, sep=" ", header=0, index=False, columns=["IDs","Speaker"])
print("done.")

stu_file = "data/test/spk2utt"
print("Writing spk2utt file: {} .. ".format(stu_file), end='')
writeSpeakerToUtterance(stu_file, test["Speaker"], test["IDs"])
print("done.")

corpus_file = "data/test/corpus.txt"
print("Writing corpus file: {} .. ".format(corpus_file), end='')
test.to_csv(corpus_file, sep="\t", header=0, index=False, columns=["Recognition"])
print("done.")

text_file = "data/test/text"
print("Writing text file: {} .. ".format(text_file), end='')
writeTextFile(text_file, test["IDs"], test["Recognition"])
print("done.")

stg_file = "data/test/spk2gender"
print("Writing spk2gender file: {} .. ".format(stg_file), end='')
writeSpeakerToGender(stg_file, test["Speaker"], gender)
print("done.")
