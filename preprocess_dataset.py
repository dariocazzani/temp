import torch
import torchaudio
import os, joblib
import librosa
import uuid
import numpy as np
from glob import glob
from tqdm import tqdm

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

if not os.path.exists(config.get("PATHS", "datapath")):
    os.mkdir(config.get("PATHS", "datapath"))

train_dataset = torchaudio.datasets.LIBRISPEECH(config.get("PATHS", "datapath"), url="train-clean-100", download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH(config.get("PATHS", "datapath"), url="test-clean", download=True)

train_labels_dict = dict()
for _, _, _, speaker_id, _, _ in tqdm(train_dataset):
    if speaker_id not in train_labels_dict.keys():
        train_labels_dict[speaker_id] = len(train_labels_dict)

joblib.dump(train_labels_dict, config.get("PATHS", "train_labels_dict"), compress=1)

# Create eval and test set
test_labels_dict = dict()
for waveform, _, _, speaker_id, _, _ in tqdm(test_dataset):
    filename = f"{uuid.uuid1().hex}.wav"
    
    if speaker_id not in test_labels_dict.keys():
        folderpath = os.path.join(config.get("PATHS", "enrollpath"), f"{speaker_id}")
        test_labels_dict[speaker_id] = None
    else:
        folderpath = os.path.join(config.get("PATHS", "testpath"), f"{speaker_id}")
    if not os.path.exists(folderpath): 
        os.makedirs(folderpath)
    librosa.output.write_wav(os.path.join(folderpath, filename), np.squeeze(waveform.cpu().numpy()), config.getint("AUDIO", "sr"))

assert(len(glob(config.get("PATHS", "enrollpath")+"/*")) == len(glob(config.get("PATHS", "testpath")+"/*"))), "Enroll and test don't have the same number of speakers"