import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import random
import torch
import joblib
from pyvad import vad
import librosa

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

from model import Model

labels_dict = joblib.load(config.get("PATHS", "train_labels_dict"))

def chunk_data(data, window_size, overlap_size=0, flatten_inside_window=True):
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1,1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros((num_windows*window_size - (num_windows-1)*overlap_size,data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = ast(data,
              shape=(num_windows,window_size*data.shape[1]),
              strides=((window_size-overlap_size)*data.shape[1]*sz,sz)
            )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows,-1,data.shape[1]))


def preprocess(x):
    audios = list()
    labels = list()
    for waveform, sample_rate, _, speaker_id, _, _ in x:
        if waveform.shape[1] < int(config.getfloat("AUDIO", "length") * config.getint("AUDIO", "sr")):
            continue # Skip short audio samples
        start = random.randint(0, max(0, waveform.shape[1] - config.getfloat("AUDIO", "length") * config.getint("AUDIO", "sr")))
        audio = waveform[:, start:start + int(config.getfloat("AUDIO", "length") * config.getint("AUDIO", "sr"))]
        audios.append(torch.squeeze(audio))
        labels.append(torch.tensor(labels_dict[speaker_id]))

    return torch.stack(audios), torch.stack(labels)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class VoiceEmbedder(object):
    def __init__(self):
        self._load()

    def _load(self):
        self.model = Model(num_classes = len(labels_dict))
        self.storage = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(config.get('PATHS', 'trained_embedder'), map_location=self.storage)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.storage)
        self.model.eval()
 
    def run(self, chunks:np.array, max_batch_size:int=256) -> list:
        """ chunks: [batch_size, audio_length] """
        output = list()
        start = 0
        end = max_batch_size
        chunks = list(chunks)
        while start < len(chunks):
            data = torch.from_numpy(np.array(chunks[start:end]).astype(np.float32)).to(self.storage)
            chunks_embeddings, _ = self.model(data)
            chunks_embeddings = list(chunks_embeddings.detach().cpu().numpy())
            output.extend(chunks_embeddings)
            start+=max_batch_size
            end+=max_batch_size
        chunks_embeddings = output
        return chunks_embeddings


def webrtc_segmentation(file_path:str, sample_rate:int, hop_length:int, aggressiveness:int) -> tuple:
    data, fs = librosa.load(file_path, mono=True, sr=sample_rate)
    data /= max(np.abs(data))
    vact = vad(data, fs, fs_vad=sample_rate, hop_length=hop_length, vad_mode=aggressiveness)
    segments = list()

    previous = 0
    for idx, pred in enumerate(vact):
        if previous == 0 and pred == 1:
            start = idx
            previous = 1
        if previous == 1 and pred == 0:
            end = idx
            previous = 0
            segments.append((start, end))
    return segments, data