
from __future__ import division

import os
import math
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument('--word_dim', dest='word_dim', type=int, help='word embedding dimension', default=20)
parser.add_argument('--hid_size', dest='hid_size', type=int, help='hidden dimension size', default=100)
parser.add_argument('--enc_size', dest='enc_size', type=int, help='encode dimension size', default=20)
parser.add_argument('--epochs', dest='epochs', type=int, help='epochs for model training', default=10)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate', default=1e-3)
parser.add_argument('--num_cate', dest='num_cate', type=int, help='categories for output', default=5)
parser.add_argument('--alpha', dest='alpha', type=float, help='ratio in different criterion', default=0.1)
parser.add_argument('--init_weight', dest='init_weight', type=float, help='initial weight', default=0.01)
parser.add_argument('--sem_com', dest='sem_com', type=bool, help='if semantic composition', default=False)
parser.add_argument('--seed', dest='seed', type=int, help='random seed', default=12345)
parser.add_argument('--emb_file', dest='emb_file', type=str, help='embedding file', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/bin/text.txt')
parser.add_argument('--data_file', dest='data_file', type=str, help='data file', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/tmp/')

args = parser.parse_args()
use_gpu = torch.cuda.is_available()

print(args)

torch.manual_seed(args.seed)

class RecursiveAutoEncoder(nn.Module):
    def __init__(self, word_dim, hid_size, enc_size, num_cate):
        super(RecursiveAutoEncoder, self).__init__()
        self.WeC2H = nn.Linear(2 * word_dim, hid_size)
        self.WeH2P = nn.Linear(hid_size, word_dim)
        self.WeP2H = nn.Linear(word_dim, hid_size)
        self.WeH2H = nn.Linear(hid_size, hid_size)
        self.WeH2E = nn.Linear(hid_size, enc_size)
        self.WeE2C = nn.Linear(enc_size, num_cate)
        self.WdC2E = nn.Linear(num_cate, enc_size)
        self.WdE2H = nn.Linear(enc_size, hid_size)
        self.WdH2H = nn.Linear(hid_size, hid_size)
        self.WdH2P = nn.Linear(hid_size, word_dim)
        self.WdP2H = nn.Linear(word_dim, hid_size)
        self.WdH2C = nn.Linear(hid_size, 2 * word_dim)
        self.tanh = nn.Tanh()
        self.soft = nn.LogSoftmax()
        
    def encoder(self, children):
        parent = self.tanh(self.WeH2P(self.tanh(self.WeC2H(children))))
        encode = self.tanh(self.WeE2C(self.tanh(self.WeH2E(self.tanh(self.WeH2H(self.tanh(self.WeP2H(parent))))))))
        scores = self.soft(encode)
        return parent, encode, scores
    
    def decoder(self, encode):
        return self.tanh(self.WdH2C(self.tanh(self.WdP2H(self.tanh(self.WdH2P(self.tanh(self.WdH2H(self.tanh(self.WdE2H(self.tanh(self.WdC2E(encode))))))))))))
    
    def forward(self, children):
        parent, encode, scores = self.encoder(children)
        decode = self.decoder(encode)
        return parent, encode, decode, scores

def set_timer(sec):
    min = math.floor(sec / 60)
    sec -= min * 60
    return '%d min %d sec!' % (min, sec)

def normalize_parent(x):
    return torch.div(x, torch.norm(x, 2))

def set_format(val):
    return '%.3f%%' % (val * 100)

def read_embedding():
    emb_voc, emb_vec = [], {}
    with open(args.emb_file, 'r') as lines:
        for line in lines:
            tokens = line.split()
            emb_voc.append(tokens[0])
            emb_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        lines.close()
    return emb_voc, emb_vec

def build_embedding(ssc_voc, ssc_vec, line, emb_voc, emb_vec):
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
            ssc_voc, ssc_vec = build_embedding(ssc_voc, ssc_vec, sents[2], emb_voc, emb_vec)
                        
    with open(os.path.join(args.data_file, 'valid.txt')) as lines:
        for line in lines:
            sents = line.lower().split('\t')
            valid_data.append(sents[0] + "\t" + sents[2] + "\t" + sents[4])
            ssc_voc, ssc_vec = build_embedding(ssc_voc, ssc_vec, sents[2], emb_voc, emb_vec)
    
    with open(os.path.join(args.data_file, 'test.txt')) as lines:
        for line in lines:
            sents = line.lower().split('\t')
            test_data.append(sents[0] + "\t" + sents[2] + "\t" + sents[4])
            ssc_voc, ssc_vec = build_embedding(ssc_voc, ssc_vec, sents[2], emb_voc, emb_vec)
                        
    return train_data, valid_data, test_data, ssc_voc, ssc_vec

def build_semcom(line, ssc_vec):
    sents = line.split('\t')
    tokens = sents[1].split()
    tags = sents[2].split()
    label = torch.LongTensor([int(sents[0])])
    
    row, idx = 0, 0
    for i in range(len(tags)):
        if tags[i][0:1] == 'b' or tags[i][0:1] == 'o':
            row += 1
    sentm = torch.FloatTensor(row, args.word_dim)
    
    for i in range(len(tags)):
        if tags[i][0:1] == 'b' or tags[i][0:1] == 'o':
            sentm[idx] = ssc_vec[tokens[i]]
            if idx < row - 1:
                idx = idx + 1 
        else:
            sentm[idx] = sentm[idx] + ssc_vec[tokens[i]]
            
    return label, sentm
    
def build_matrix(line, ssc_vec):
    sents = line.split('\t')
    tokens = sents[1].split()
    label = torch.LongTensor([int(sents[0])])
    sentm = torch.FloatTensor(len(tokens), args.word_dim).zero_()
    
    for i in range(len(tokens)):
        sentm[i] = ssc_vec[tokens[i]]
    
    return label, sentm  
    
rae = RecursiveAutoEncoder(args.word_dim, args.hid_size, args.enc_size, args.num_cate)

mse_criterion = nn.MSELoss(size_average=False)
nll_criterion = nn.NLLLoss()
optimizer = optim.Adam(rae.parameters(), lr=args.learning_rate)

if use_gpu:
    rae = rae.cuda()
    mse_criterion = mse_criterion.cuda()
    nll_criterion = nll_criterion.cuda()
print(rae)
loss_storage = []
accu_storage = []

def train():
    emb_voc, emb_vec = read_embedding()
    train_data, valid_data, test_data, ssc_voc, ssc_vec = read_corpus(emb_voc, emb_vec)    
    
    best_train, best_valid = 0, 0
    for i in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_correct = 0, 0
        random.shuffle(train_data)
        random.shuffle(valid_data)
        for j in range(len(train_data)):
            if args.sem_com:
                label, sentm = build_semcom(train_data[j], ssc_vec)
            else:
                label, sentm = build_matrix(train_data[j], ssc_vec) 
            if use_gpu:
                label = label.cuda()
                sentm = sentm.cuda()   
            parent = sentm[0]    
            for k in range(1, len(sentm)):
                for l in range(args.epochs):
                    children = Variable((torch.cat((sentm[k], parent), 0)).view(1, -1))
                    parent, encode, decode, scores = rae(children)                
                    loss = nll_criterion(scores, Variable(label))#mse_criterion(decode, children) + 
                    if k == len(sentm) - 1:  
                        #loss = loss + nll_criterion(scores, Variable(label))                               
                        train_loss = train_loss + loss.data[0] 
                        _, predicted = torch.max(scores.data, 1)
                        if predicted[0] == label[0]:
                            train_correct += 1 
                    optimizer.zero_grad()
                    loss.backward()    
                    optimizer.step()
                    parent = Variable(normalize_parent(parent.data)).data.view(args.word_dim, -1)
        loss_storage.append(train_loss / len(train_data) / args.epochs)  
        accu_storage.append(train_correct / len(train_data)) 
        if (train_correct / len(train_data)) > best_train:
            best_train = (train_correct / len(train_data))
     
        valid_correct = 0
        for j in range(len(valid_data)):
            if args.sem_com:
                label, sentm = build_semcom(valid_data[j], ssc_vec)
            else:
                label, sentm = build_matrix(valid_data[j], ssc_vec) 
            if use_gpu:
                label = label.cuda()
                sentm = sentm.cuda()   
            parent = sentm[0]
            for k in range(1, len(sentm)):
                children = Variable((torch.cat((sentm[k], parent), 0)).view(1, -1))
                parent, encode, scores = rae.encoder(children)
                if k == len(sentm) - 1:
                    _, predicted = torch.max(scores.data, 1)
                    if predicted[0] == label[0]:
                        valid_correct += 1 
                parent = Variable(normalize_parent(parent.data)).data.view(args.word_dim, -1)
        end = time.time()
        print("Epoch: " + str(i) + "\tTrain loss: " + '%.6f' % (train_loss / len(train_data)) + "\tTrain accuracy: " + set_format(train_correct / len(train_data)) + "\t\tValid accuracy: " + set_format(valid_correct / len(valid_data)) + "\t\tCost time: " + set_timer(end - start))  
        if (valid_correct / len(valid_data)) > best_valid:
            best_valid = (valid_correct / len(valid_data))
         
    start = time.time()    
    test_correct = 0
    random.shuffle(test_data)
    for j in range(len(test_data)):
        if args.sem_com:
            label, sentm = build_semcom(test_data[j], ssc_vec)
        else:
            label, sentm = build_matrix(test_data[j], ssc_vec) 
        if use_gpu:
            label = label.cuda()
            sentm = sentm.cuda()   
        parent = sentm[0]
        for k in range(1, len(sentm)):
            children = Variable((torch.cat((sentm[k], parent), 0)).view(1, -1))
            parent, encode, scores = rae.encoder(children) 
            if k == len(sentm) - 1:
                _, predicted = torch.max(scores.data, 1)
                if predicted[0] == label[0]:
                    test_correct += 1 
            parent = Variable(normalize_parent(parent.data)).data.view(args.word_dim, -1) 
    end = time.time()  
    print("Test accuracy: " + set_format(test_correct / len(test_data)) + "\tCost time: " + set_timer(end - start)) 
    print("Best train: " + set_format(best_train) + "\tBest valid: " + set_format(best_valid))                   

def main():
    start = time.time()
    train()
    end = time.time()
    if args.sem_com:
        print("The model cost " + set_timer(end - start) + " for training with semantic composition pattern.")
    else:
        print("The model cost " + set_timer(end - start) + " for training with single word embedding pattern.")
    plt.plot(range(args.epochs), loss_storage, label="loss", color="red")
    plt.plot(range(args.epochs), accu_storage, label="accuracy", color="green")
    plt.legend()
    plt.show() 
         
if __name__ == "__main__":
    main()
