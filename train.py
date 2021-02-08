from __future__ import print_function
from models import *

from util import Dictionary

import torch
import torch.nn as nn
import torch.optim as optim

import json
import time
import random
import os
import hydra
from omegaconf import DictConfig

@hydra.main(config_name="config")
def get_config(cfg : DictConfig):
    return cfg

def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')

def package(data):
    """Package data for training / evaluation."""
    data = list(map(lambda x: json.loads(x), data))
    dat = list(map(lambda x: list(map(lambda y: dictionary.word2idx[y], x['text'])), data))
    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    targets = list(map(lambda x: x['label'], data))
    maxlen = min(maxlen, 500)
    for i in range(len(data)):
        if maxlen < len(dat[i]):
            dat[i] = dat[i][:maxlen]
        else:
            for j in range(maxlen - len(dat[i])):
                dat[i].append(dictionary.word2idx['<pad>'])
    dat = torch.LongTensor(dat)
    targets = torch.LongTensor(targets)
    return dat.t(), targets

@torch.no_grad()
def evaluate(cfg):
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    for batch, i in enumerate(range(0, len(data_val), cfg.training.batch_size)):
        data, targets = package(data_val[i:min(len(data_val), i+cfg.training.batch_size)])
        if cfg.cuda:
            data = data.cuda()
            targets = targets.cuda()
        hidden = model.init_hidden(data.size(1))
        output, attention, _ = model.forward(data, hidden)
        output_flat = output.view(data.size(1), -1)
        total_loss += criterion(output_flat, targets).data
        prediction = torch.max(output_flat, 1)[1]
        total_correct += torch.sum((prediction == targets).float())
    return total_loss.item() / (len(data_val) // cfg.training.batch_size), total_correct.data.item() / len(data_val)


def train(epoch_number ,cfg):
    global best_val_loss, best_acc
    model.train()
    total_loss = 0
    total_pure_loss = 0  # without the penalization term
    start_time = time.time()
    for batch, i in enumerate(range(0, len(data_train), cfg.training.batch_size)):
        data, targets = package(data_train[i:i+cfg.training.batch_size])
        if cfg.cuda:
            data = data.cuda()
            targets = targets.cuda()
        hidden = model.init_hidden(data.size(1))
        output, attention, out_bool = model.forward(data, hidden)
        loss = criterion(output.view(data.size(1), -1), targets)
        total_pure_loss += loss.data.item()
        if out_bool:  # add penalization term
            attentionT = torch.transpose(attention, 1, 2).contiguous()
            extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
            loss += cfg.training.penalization_coeff * extra_loss
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), cfg.model.clip)
        optimizer.step()

        total_loss += loss.data.item()

        if batch % cfg.training.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f} | pure loss {:5.4f}'.format(
                  epoch_number, batch, len(data_train) // cfg.training.batch_size,
                  elapsed * 1000 / cfg.training.log_interval, total_loss / cfg.training.log_interval,
                  total_pure_loss / cfg.training.log_interval))
            total_loss = 0
            total_pure_loss = 0
            start_time = time.time()

#            for item in model.parameters():
#                print item.size(), torch.sum(item.data ** 2), torch.sum(item.grad ** 2).data[0]
#            print model.encoder.ws2.weight.grad.data
#            exit()
    evaluate_start_time = time.time()
    val_loss, acc = evaluate(cfg)
    print('-' * 89)
    fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
    print(fmt.format((time.time() - evaluate_start_time), val_loss, acc))
    print('-' * 89)
    # Save the model, if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(cfg.data.save, 'wb') as f:
            torch.save(model, f)
        f.close()
        best_val_loss = val_loss
    else:  # if loss doesn't go down, divide the learning rate by 5.
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
    if not best_acc or acc > best_acc:
        with open(cfg.data.save[:-3]+'.best_acc.pt', 'wb') as f:
            torch.save(model, f)
        f.close()
        best_acc = acc
    with open(cfg.data.save[:-3]+'.epoch-{:02d}.pt'.format(epoch_number), 'wb') as f:
        torch.save(model, f)
    f.close()


@hydra.main(config_name="config")
def main(cfg):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        if not cfg.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # Load Dictionary
    assert os.path.exists(cfg.data.train_data)
    assert os.path.exists(cfg.data.val_data)
    print('Begin to load the dictionary.')
    global dictionary
    dictionary = Dictionary(path=cfg.data.dictionary)

    global best_val_loss
    global best_acc
    best_val_loss = None
    best_acc = None

    n_token = len(dictionary)
    
    global model 
    model = Classifier({
        'dropout': cfg.model.dropout,
        'ntoken': n_token,
        'nlayers': cfg.model.nlayers,
        'nhid': cfg.model.nhid,
        'ninp': cfg.model.emsize,
        'pooling': 'all',
        'attention-unit': cfg.model.attention_unit,
        'attention-hops': cfg.model.attention_hops,
        'nfc': cfg.model.nfc,
        'dictionary': dictionary,
        'word-vector': cfg.data.word_vector,
        'class-number': cfg.class_number
    })
    if cfg.cuda:
        model = model.cuda()

    global I
    I = torch.zeros(cfg.training.batch_size, cfg.model.attention_hops, cfg.model.attention_hops)
    for i in range(cfg.training.batch_size):
        for j in range(cfg.model.attention_hops):
            I.data[i][j][j] = 1
    if cfg.cuda:
        I = I.cuda()

    global criterion 
    global optimizer
    criterion = nn.CrossEntropyLoss()
    if cfg.training.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    elif cfg.training.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr, momentum=0.9, weight_decay=0.01)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')
    print('Begin to load data.')
    global data_train 
    data_train = open(cfg.data.train_data).readlines()
    global data_val 
    data_val = open(cfg.data.val_data).readlines()
    try:
        for epoch in range(cfg.training.epochs):
            train(epoch, cfg)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exit from training early.')
        data_val = open(cfg.data.test_data).readlines()
        evaluate_start_time = time.time()
        test_loss, acc = evaluate(cfg)
        print('-' * 89)
        fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
        print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
        print('-' * 89)
        exit(0)

if __name__ == '__main__':
    main()