import torch
import torchaudio
import os, joblib
import uuid
import json
import numpy as np
from glob import glob
from tqdm import tqdm

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

if not os.path.exists(config.get("PATHS", "datapath")):
    os.mkdir(config.get("PATHS", "datapath"))

trainsets = json.loads(config.get("DATA", "trainsets"))

train_labels_dict = dict()
for trainset in trainsets:
    train_dataset = torchaudio.datasets.LIBRISPEECH(config.get("PATHS", "datapath"), url=trainset, download=True)
    print(f"Preprocessing: {trainset}")
    for _, _, _, speaker_id, _, _ in tqdm(train_dataset):
        if speaker_id not in train_labels_dict.keys():
            train_labels_dict[speaker_id] = len(train_labels_dict)

joblib.dump(train_labels_dict, config.get("PATHS", "train_labels_dict"), compress=1)