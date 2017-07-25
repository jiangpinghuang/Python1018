import os
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument('--word_dim', dest='word_dim', type=int, help='word embedding dimension', default=20)
parser.add_argument('--hid_size', dest='hid_size', type=int, help='hidden dimension size', default=50)
parser.add_argument('--enc_size', dest='enc_size', type=int, help='encode dimension size', default=10)
parser.add_argument('--epochs', dest='epochs', type=int, help='epochs for model training', default=30)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate', default=1e-3)
parser.add_argument('--num_class', dest='num_class', type=int, help='class for output', default=5)
parser.add_argument('--init_weight', dest='init_weight', type=float, help='initial weight', default=0.1)
parser.add_argument('--sem_com', dest='sem_com', type=bool, help='if semantic composition', default=True)
parser.add_argument('--seed', dest='seed', type=long, help='random seed', default=2718281828459045232536)
parser.add_argument('--emb_file', dest='emb_file', type=str, help='embedding file', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/bin/text.txt')
parser.add_argument('--data_file', dest='data_file', type=str, help='data file', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/ssc/')

args = parser.parse_args()

print(args)

torch.manual_seed(args.seed)

class RecursiveAutoEncoder(nn.Module):
    def __init__(self, word_dim, hid_size, enc_size):
        super(RecursiveAutoEncoder, self).__init__()
        self.WeC2P = nn.Linear(2 * word_dim, word_dim)
        self.WeP2H = nn.Linear(word_dim, hid_size)
        self.WeH2H = nn.Linear(hid_size, hid_size)
        self.WeH2E = nn.Linear(hid_size, enc_size)
        self.WdE2H = nn.Linear(enc_size, hid_size)
        self.WdH2H = nn.Linear(hid_size, hid_size)
        self.WdH2P = nn.Linear(hid_size, word_dim)
        self.WdP2C = nn.Linear(word_dim, 2 * word_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def encoder(self, children):
        parent = self.tanh(self.WeC2P(children))
        return parent, self.tanh(self.WeH2E(self.tanh(self.WeH2H(self.tanh(self.WeP2H(parent))))))
    
    def decoder(self, encoding):
        return self.tanh(self.WdP2C(self.tanh(self.WdH2P(self.tanh(self.WdH2H(self.tanh(self.WdE2H(encoding))))))))
    
rae = RecursiveAutoEncoder(args.word_dim, args.hid_size, args.enc_size)
print(rae)

def normalize_parent(x):
    return torch.div(x, torch.norm(x, 2))

def read_embedding():
    emb_voc = []
    emb_vec = {}
    with open(args.emb_file, 'r') as lines:
        for line in lines:
            tokens = line.split()
            emb_voc.append(tokens[0])
            emb_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        lines.close()
    return emb_voc, emb_vec

def build_embedding(ssc_voc, ssc_vec, line):
    tokens = line.split()
    for i in range(len(tokens)):
        if tokens[i] not in ssc_voc:
            ssc_voc.append(tokens[i])
            if tokens[i] in emb_voc:
                ssc_vec[tokens[i]] = torch.from_numpy(emb_vec[tokens[i]]).view(1, args.word_dim)
            else:
                ssc_vec[tokens[i]] = init.normal(torch.Tensor(1, args.word_dim), 0, args.init_weight) 
    return ssc_voc, ssc_vec   

def read_corpus(emb_voc, emb_vec):
    ssc_voc, ssc_vec = [], {}
    train_data, valid_data, test_data = [], [], []    
    assert os.path.exists(args.data_file)
    
    with open(os.path.join(args.data_file, 'train.txt')) as lines:
        for line in lines:
            sents = line.lower().split('\t')
            train_data.append(sents[0] + "\t" + sents[2] + "\t" + sents[4])
            ssc_voc, ssc_vec = build_embedding(ssc_voc, ssc_vec, sents[2])
                        
    with open(os.path.join(args.data_file, 'valid.txt')) as lines:
        for line in lines:
            sents = line.lower().split('\t')
            valid_data.append(sents[0] + "\t" + sents[2] + "\t" + sents[4])
            ssc_voc, ssc_vec = build_embedding(ssc_voc, ssc_vec, sents[2])
    
    with open(os.path.join(args.data_file, 'test.txt')) as lines:
        for line in lines:
            sents = line.lower().split('\t')
            test_data.append(sents[0] + "\t" + sents[2] + "\t" + sents[4])
            ssc_voc, ssc_vec = build_embedding(ssc_voc, ssc_vec, sents[2])
                        
    return train_data, valid_data, test_data, ssc_voc, ssc_vec


