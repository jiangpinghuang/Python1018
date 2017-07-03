# -*- coding utf-8 -*-
"""
@date: Created on Sun Jun 25 06:20:18 2017.
@description: There is a recursive neural network based on semantic composition for textual similarity.
@author: Jiangping Huang (hjp@whu.edu.cn).
@copyright: All right reserved.
@license: The code is released under the Apache License.
@version: Alpha.
"""

import os
import math
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument('--word_dim', dest='word_dim', type=int, help='embedding dimension', default=20)
parser.add_argument('--hid_dim', dest='hid_dim', type=int, help='hidden dimension', default=50)
parser.add_argument('--epochs', dest='epochs', type=int, help='epochs for model training', default=10)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate', default=1e-3)
parser.add_argument('--num_class', dest='num_class', type=int, help='classes of output', default=2)
parser.add_argument('--sem_com', dest='sem_com', type=bool, help='semantic composition', default=True)
parser.add_argument('--seed', dest='seed', type=int, help='random seed', default=1234567890)
parser.add_argument('--emb_file', dest='emb_file', type=str, help='embedding file path', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/bin/text.txt')
parser.add_argument('--data_file', dest='data_file', type=str, help='data file path', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/tmp/')

use_gpu = torch.cuda.is_available()

args = parser.parse_args()
torch.manual_seed(args.seed)

class RvNN(nn.Module):
    def __init__(self, word_dim, hid_size, num_class):
        super(RvNN, self).__init__()
        self.linear1 = nn.Linear(2 * word_dim, word_dim)
        self.linear2 = nn.Linear(word_dim, word_dim)
        self.linear3 = nn.Linear(2 * word_dim, hid_size)
        self.linear4 = nn.Linear(hid_size, hid_size / 10)
        self.linear5 = nn.Linear(hid_size / 10, num_class)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
        
    def forward(self, left, right, word_dim):
        c1 = Variable(torch.FloatTensor(torch.zeros(word_dim)))
        c2 = Variable(torch.FloatTensor(torch.zeros(word_dim)))
        
        for i in range(len(left) - 1):
            if i == 0:
                c1 = self.linear1((torch.cat((left[i], left[i + 1]), 0)).view(1, -1))
                #c1 = self.relu(self.linear2(c1))
                c1 = self.linear2(c1)
            else:
                c1 = self.linear1((torch.cat((c1.view(word_dim, -1), left[i + 1]), 0)).view(1, -1))
                #c1 = self.relu(self.linear2(c1))
                c1 = self.linear2(c1)
                
        for j in range(len(right) - 1):
            if j == 0:
                c2 = self.linear1((torch.cat((right[j], right[j + 1]), 0)).view(1, -1))
                #c2 = self.relu(self.linear2(c2))
                c2 = self.linear2(c2)
            else:
                c2 = self.linear1((torch.cat((c2.view(word_dim, -1), right[j + 1]), 0)).view(1, -1))
                #c2 = self.relu(self.linear2(c2))
                c2 = self.linear2(c2)
        
        print("c1: ", c1)
        print("c2: ", c2)        
        concat = self.relu(self.linear3(torch.cat((c1, c2), -1)))
        print("concat: ", concat)
        output = self.softmax(self.linear5(self.relu(self.linear4(concat))))  
        
        return output      

def set_timer(sec):
    min = math.floor(sec / 60)
    sec -= min * 60
    return '%dm %ds' % (min, sec)

def build_vector(path):
    vector = {}
    vocab = []

    with open(path, 'r') as lines:
        for line in lines:
            tokens = line.split()
            vocab.append(tokens[0])
            vector[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        lines.close()  
          
    return vector, vocab

def build_corpus(path):
    trainSet = []
    testSet = []
    
    assert os.path.exists(path)
    
    train = os.path.join(path, 'train.txt')
    test = os.path.join(path, 'test.txt')
    
    with open(train, 'r') as lines:
        for line in lines:
            sents = line.lower().split('\t')
            trainSet.append(sents[0] + "\t" + sents[2] + "\t" + sents[6] + "\t" + sents[4] + "\t" + sents[8])
            
    with open(test, 'r') as lines:
        for line in lines:
            sents = line.lower().split('\t')
            testSet.append(sents[0] + "\t" + sents[2] + "\t" + sents[6] + "\t" + sents[4] + "\t" + sents[8])
    
    return trainSet, testSet

def build_semcom(line, vector, vocab):
    sents = line.split('\t')
    label = torch.LongTensor(1, 1)
    label[0] = int(sents[0])
    
    lsent = sents[1].split()
    rsent = sents[2].split()
    ltags = sents[3].split()
    rtags = sents[4].split()
    
    lrow, rrow = 0, 0
    for i in range(len(ltags)):
        if ltags[i][0:1] == 'b' or ltags[i][0:1] == 'o':
            lrow += 1
    for j in range(len(rtags)):
        if rtags[j][0:1] == 'b' or rtags[j][0:1] == 'o':
            rrow += 1
    lsentm = torch.FloatTensor(lrow, args.word_dim)
    rsentm = torch.FloatTensor(rrow, args.word_dim)
    
    lidx, ridx = 0, 0
    for i in range(len(ltags)):
        if ltags[i][0:1] == 'b' or ltags[i][0:1] == 'o':
            if lsent[i] in vocab:
                lsentm[lidx] = torch.from_numpy(vector[lsent[i]]).view(1, args.word_dim)
            else:
                wv = torch.Tensor(1, args.word_dim)
                lsentm[lidx] = init.normal(wv, 0, 0.1)
            if lidx < lrow - 1:
                lidx = lidx + 1
        else:
            if lsent[i] in vocab:
                lsentm[lidx] = lsentm[lidx] + torch.from_numpy(vector[lsent[i]]).view(1, args.word_dim)
            else:
                wv = torch.Tensor(1, args.word_dim)
                lsentm[lidx] = lsentm[lidx] + init.normal(wv, 0, 0.1)
                
    for j in range(len(rtags)):
        if rtags[j][0:1] == 'b' or rtags[j][0:1] == 'o':
            if rsent[j] in vocab:
                rsentm[ridx] = torch.from_numpy(vector[rsent[j]]).view(1, args.word_dim)
            else:
                wv = torch.Tensor(1, args.word_dim)
                rsentm[ridx] = init.normal(wv, 0, 0.1)
            if ridx < rrow - 1:
                ridx = ridx + 1
        else:
            if rsent[j] in vocab:
                rsentm[ridx] = rsentm[ridx] + torch.from_numpy(vector[rsent[j]]).view(1, args.word_dim)
            else:
                wv = torch.Tensor(1, args.word_dim)
                rsentm[ridx] = rsentm[ridx] + init.normal(wv, 0, 0.1)
    
    return label, lsentm, rsentm    

def build_matrix(line, vector, vocab):
    sents = line.split('\t')
    lsent = sents[1].split()
    rsent = sents[2].split()
    
    label = torch.LongTensor(1, 1)
    lsentm = torch.FloatTensor(len(lsent), args.word_dim)
    rsentm = torch.FloatTensor(len(rsent), args.word_dim)
    
    label[0] = int(sents[0])
    
    for i in range(len(lsent)):
        if lsent[i] in vocab:
            lsentm[i] = torch.from_numpy(vector[lsent[i]]).view(1, args.word_dim)
        else:
            wv = torch.Tensor(1, args.word_dim)
            lsentm[i] = init.normal(wv, 0, 0.1)
            
    for j in range(len(rsent)):
        if rsent[j] in vocab:
            rsentm[j] = torch.from_numpy(vector[rsent[j]]).view(1, args.word_dim)
        else:
            wv = torch.Tensor(1, args.word_dim)
            rsentm[j] = init.normal(wv, 0, 0.1)
            
    return label, lsentm, rsentm

def train(trainSet, vector, vocab, rnn):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(rnn.parameters(), lr=args.learning_rate)
    
    for i in range(args.epochs):
        start = time.time()
        for j in range(len(trainSet)):
            if args.sem_com:
                label, lsentm, rsentm = build_semcom(trainSet[j], vector, vocab)
            else:
                label, lsentm, rsentm = build_matrix(trainSet[j], vector, vocab)
                
            if use_gpu:
                label = Variable(label[0].cuda())
                lsentm = Variable(lsentm.cuda())
                rsentm = Variable(rsentm.cuda())
            else:
                label = Variable(label[0])
                lsentm = Variable(lsentm)
                rsentm = Variable(rsentm)
            
            optimizer.zero_grad()
            pred = rnn(lsentm, rsentm, args.word_dim)
            print("The epoch %d " % (i+1) + "and %d example." % (j+1))
            print(pred.data)
            
            loss = criterion(pred, label)
            print(loss.data)
            loss.backward()
            optimizer.step()
            
        end = time.time()
        print("The epoch %d cost " % (i + 1) + set_timer(end - start))    
    
def test(testSet, vector, vocab):
    for i in range(len(testSet)):
        print(testSet[i])        

if __name__ == "__main__":
    print(args)
    start = time.time()
    
    vector, vocab = build_vector(args.emb_file)
    trainSet, testSet = build_corpus(args.data_file)
    
    random.shuffle(trainSet)
    random.shuffle(testSet)
    
    rnn = RvNN(args.word_dim, args.hid_dim, args.num_class)
    print(rnn)    
    
    if use_gpu:
        rnn = rnn.cuda()
        
    train(trainSet, vector, vocab, rnn)
    
    end = time.time()
    print("The model cost " + set_timer(end - start) + " for training.")

