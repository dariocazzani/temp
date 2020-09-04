import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from online_triplet_loss.losses import batch_hard_triplet_loss

import random
from tqdm import tqdm
import joblib
import multiprocessing as mp
import time
import numpy as np
from glob import glob
import librosa

from model import Model
from utils import chunk_data, preprocess, AverageMeter, accuracy

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

labels_dict = joblib.load(config.get("PATHS", "train_labels_dict"))

from sklearn.neighbors import KNeighborsClassifier
enroll_files = glob(config.get("PATHS", "enrollpath") + "/**/*wav", recursive=True)
test_files = glob(config.get("PATHS", "enrollpath") + "/**/*wav", recursive=True)

print(f"Num test speakers: {len(set([x.split('/')[-2] for x in test_files]))}")


def main():
    train_dataset = torchaudio.datasets.LIBRISPEECH(config.get("PATHS", "datapath"), url="train-clean-100", download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(config.get("PATHS", "datapath"), url="test-clean", download=True)

    kwargs = {'num_workers': int(mp.cpu_count()), 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=config.getint("HYPERPARAMS", "batch_size"),
                                    shuffle=True,
                                    collate_fn=lambda x: preprocess(x),
                                    drop_last=True,
                                    **kwargs)

    torch.manual_seed(config.get("SEED", "value"))
    model = Model(num_classes = len(labels_dict))
    device = torch.device("cuda:0")

    print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()]):,}')
    print(f'Number of classes: {len(labels_dict)}')

    ce_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD([{'params': model.parameters()}],
                                lr=config.getfloat("HYPERPARAMS", "lr"),
                                momentum=config.getfloat("HYPERPARAMS", "momentum"),
                                weight_decay=config.getfloat("HYPERPARAMS", "weigth_decay"))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    start_epoch = 0
    model.to(device)
    best_acc = 0

    try:
        for epoch in range(start_epoch, config.getint("HYPERPARAMS", "epochs")):
            # train for one epoch
            now = time.time()
            train(train_loader, model, ce_criterion, optimizer, epoch, device, best_acc)

            # evaluate on validation set
            val_acc = validate(model, device)
            print(f"Current validation accuracy: {val_acc:.2f}%")

            scheduler.step(val_acc)

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)

            print(f'Best accuracy: {best_acc:.2f}%')
            print("Ran epoch {} in {:.1f} seconds".format(epoch, time.time()-now))
            
            # Save model with highest validation accuracy
            if is_best:
                best_val_prec_path = config.get("PATHS", "trained_embedder")
                print(f"Saving new best validation accuracy model to {best_val_prec_path}")

                model_state_dict = model.state_dict() 
                torch.save({
                    'best_acc': best_acc,
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'scheduler': scheduler,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, best_val_prec_path)

    except KeyboardInterrupt:
        print("Manual interrupt")


def train(train_loader, model, ce_criterion, optimizer, epoch, device, best_acc):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    ce_losses = AverageMeter()
    triplet_losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, (x, target) in enumerate(train_loader):
        x = x.to(device)
        target = target.to(device)
        input_var = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(target)

        features, output = model(input_var, train=True)
        ce_loss = ce_criterion(output, target_var)
        triplet_loss = batch_hard_triplet_loss(target_var, features, margin=1, device=device)
        loss = ce_loss + config.getfloat("HYPERPARAMS", "triplet_loss_alpha") * triplet_loss

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        
        top1.update(prec1.item(), x.size(0))
        ce_losses.update(ce_loss.data.item(), x.size(0))
        triplet_losses.update(triplet_loss.data.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % config.getint("MISC", "log_interval") == 0:
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'CE Loss {ce_loss.value:.4f} ({ce_loss.ave:.4f})\t'
                      'Triplet Loss {triplet_loss.value:.4f} ({triplet_loss.ave:.4f})\t'
                      'Accuracy {top1.value:.2f}% ({top1.ave:.2f}%)\t'.format(
                       epoch, batch_idx+1, train_batches_num, batch_time=batch_time,
                       ce_loss=ce_losses, triplet_loss=triplet_losses, top1=top1))

            print(string)
   
def validate(model, device) -> float:
    # switch to evaluate mode
    model.eval()
    X = list()
    y = list()
    print(f"Computing enrollment vectors...")
    for en_file in tqdm(enroll_files):
        audio, sr = librosa.load(en_file)
        chunks = chunk_data(audio, int(config.getfloat("AUDIO", "length") * config.getint("AUDIO", "sr")))
        chunks = torch.from_numpy(chunks.astype(np.float32)).to(device)
        embeddings, _ = model(chunks, train=False)
        embeddings = list(embeddings.cpu().detach().numpy())
        X.extend(embeddings)
        label = en_file.split("/")[-2]
        y.extend([label]*len(embeddings))

    print(f"Creating Nearest Neighbor classifier")
    neigh = KNeighborsClassifier(n_neighbors=3, metric='cosine')
    neigh.fit(X, y)

    correct = 0
    wrong = 0
    print(f"Running validation...")
    for t_file in tqdm(test_files):
        audio, sr = librosa.load(t_file)
        chunks = chunk_data(audio, int(config.getfloat("AUDIO", "length") * config.getint("AUDIO", "sr")))
        chunks = torch.from_numpy(chunks.astype(np.float32)).to(device)
        embeddings, _ = model(chunks, train=False)
        embeddings = list(embeddings.cpu().detach().numpy())
        label = t_file.split("/")[-2]
        for embedding in embeddings:
            out = neigh.predict(embedding[None, :])
            if out[0] == label:
                correct += 1
            else:
                wrong += 1

    return 100 * (correct / (correct+wrong))

if __name__ == "__main__":
    main()