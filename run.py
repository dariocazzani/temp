import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import joblib
import multiprocessing as mp

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")


def preprocess(x):
    audios = list()
    labels = list()
    for waveform, sample_rate, _, speaker_id, _, _ in x:
        start = random.randint(0, waveform.shape[1] - config.getfloat("AUDIO", "length") * config.getint("AUDIO", "sr"))
        audio = waveform[:, start:start + int(config.getfloat("AUDIO", "length") * config.getint("AUDIO", "sr"))]
        audios.append(torch.squeeze(audio))
        labels.append(torch.tensor(labels_dict[speaker_id]))

    return torch.stack(audios), torch.stack(labels)


def main():
    train_dataset = torchaudio.datasets.LIBRISPEECH(config.get("PATHS", "datapath"), url="train-clean-100", download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(config.get("PATHS", "datapath"), url="test-clean", download=True)

    labels_dict = joblib.load(config.get("PATHS", "train_labels_dict"))
    kwargs = {'num_workers': int(mp.cpu_count()), 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=config.getint("HYPERPARAMS", "batch_size"),
                                    shuffle=True,
                                    collate_fn=lambda x: preprocess(x),
                                    **kwargs)

    for audio, label in tqdm(train_loader):
        # print(label)
        pass

if __name__ == "__main__":
    main()