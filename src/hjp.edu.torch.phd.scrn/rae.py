# '''
# Created on July 15, 2017
# 
# @author: hjp
# '''
import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument('--word_dim', dest='word_dim', type=int, help='embedding dimension', default=20)
parser.add_argument('--hid_size', dest='hid_size', type=int, help='hidden dimension size', default=50)
parser.add_argument('--epochs', dest='epochs', type=int, help='epochs for model training', default=10)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate', default=1e-3)
parser.add_argument('--num_class', dest='num_class', type=int, help='classes of output', default=5)
parser.add_argument('--sem_com', dest='sem_com', type=bool, help='semantic composition', default=True)
parser.add_argument('--seed', dest='seed', type=int, help='random seed', default=271828182845904523536)
parser.add_argument('--emb_file', dest='emb_file', type=str, help='embedding file path', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/bin/text.txt')
parser.add_argument('--data_file', dest='data_file', type=str, help='data file path', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/ssc/')
 
args = parser.parse_args()

torch.manual_seed(args.seed)

class RAE(nn.Module):
    def __init__(self, word_dim, hid_size):
        super(RAE, self).__init__()
        self.Wec = nn.Linear(2 * word_dim, word_dim)
        self.Weh = nn.Linear(word_dim, hid_size)
        self.Whd = nn.Linear(hid_size, word_dim)
        self.Wdc = nn.Linear(word_dim, 2 * word_dim)
        self.tanh = nn.Tanh()
        
    def encoder(self, chd):
        wec = self.tanh(self.Wec(chd))
        enc = self.tanh(self.Weh(wec))
        return wec, enc
    
    def decoder(self, enc):
        whd = self.tanh(self.Whd(enc))
        wdc = self.tanh(self.Wdc(whd))
        return wdc
    
    def forward(self, chd):
        wec, enc = self.encoder(chd)
        wdc = self.decoder(enc)
        return wec, enc, wdc
    
rae = RAE(args.word_dim, args.hid_size)
criterion = nn.MSELoss(size_average=False)
optimizer = optim.SGD(rae.parameters(), lr=args.learning_rate)

def norm(x):
    return torch.div(x, torch.norm(x, 2))

def cos_sim(x, y):
    xx, yy, xy = 0, 0, 0
    for i in range(len(x[0])):
        xx = xx + x[0][i] ** 2
        yy = yy + y[0][i] ** 2
        xy = xy + x[0][i] * y[0][i]
    return xy / math.sqrt(xx * yy)

def euc_sim(x, y):
    xy = 0
    for i in range(len(x[0])):
        xy = xy + (y[0][i] - x[0][i]) ** 2
    return math.sqrt(xy)

def build_vector(emb_file):
    vocab = []
    vector = {}     
    with open(emb_file, 'r') as lines:
        for line in lines:
            tokens = line.split()
            vocab.append(tokens[0])
            vector[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        lines.close()           
    return vocab, vector

def build_ssc_vector(line, vocab, vector):
    lines = (line.lower()).split('\t')

    sents = lines[1].split()
    token = lines[2].split()

    label = torch.LongTensor([int(lines[0])])    
    sentm = torch.FloatTensor(len(sents), args.word_dim)
    
    for i in range(len(sents)):
        if sents[i] in vocab:
            sentm[i] = torch.from_numpy(vector[lsent[i]]).view(1, args.word_dim)
        else:
            wv = torch.Tensor(1, args.word_dim)
            sentm[i] = init.normal(wv, 0, 0.1)
    
    return label, sentm

def build_corpus():
    train_data = []
    valid_data = []
    test_data = []
    
    assert os.path.exists(args.data_file)
    
    with open(os.path.join(args.data_file, 'train.txt')) as lines:
        for line in lines:
            sents = line.lower().split('\t')
            train_data.append(sents[0] + "\t" + sents[2] + "\t" + sents[6])
            
    with open(os.path.join(args.data_file, 'valid.txt')) as lines:
        for line in lines:
            sents = line.lower().split('\t')
            valid_data.append(sents[0] + "\t" + sents[2] + "\t" + sents[6])
    
    with open(os.path.join(args.data_file, 'test.txt')) as lines:
        for line in lines:
            sents = line.lower().split('\t')
            test_data.append(sents[0] + "\t" + sents[2] + "\t" + sents[6])
    
    return train_data, valid_data, test_data
        
def train(train, valid, vocab, vector): 
    for i in range(len(train)):
        print train[i]

def valid():
    x = x
    
def test():
    x = x

def main():
    vocab, vector = build_vector(args.emb_file)
    build_corpus()
    #train_data, valid_data, test_data = build_corpus()    
    
    #train(train, vocab, vector)

    
if __name__ == "__main__":
    main()     
    
# 
# 
# 
# 
# 
# 
# 
# 
# 
# input = torch.randn(word_num, word_dim)
# 
# #print(rae)
# #print(input)
# 
# def VecNorm(x):
#     value = torch.norm(x, len(x.data[0]))
#     x = Variable(torch.div(x.data[0], value))
#     return x
# 
# def CosSim(x, y):
#     sumxx, sumxy, sumyy = 0, 0, 0
#     for i in range(len(x[0])):
#         sumxx = sumxx + x[0][i] ** 2
#         sumyy = sumyy + y[0][i] ** 2
#         sumxy = sumxy + x[0][i] * y[0][i]
#     score = sumxy / math.sqrt(sumxx * sumyy)    
#     return score
# 
# def EucSim(x, y):
#     sum = 0
#     for i in range(len(x[0])):
#         sum = sum + (y[0][i] - x[0][i]) ** 2
#     score = math.sqrt(sum)
#     return score 
# 
# loss1 = []
# losscos = []
# losseuc = []
# 
# 
# 
# def trainDemo():
#     for j in range(1000):
#         wec = input[0]
#         for i in range(1, word_num):
#             chd = Variable((torch.cat((input[i], wec), 0)).view(1, -1))
#             wec, enc, wdc = rae(chd)
#             loss = criterion(wdc, chd)
#             optimizer.zero_grad()
#             loss.backward()
# 
#             if i == 10 and j % 10 == 0:
#                 print(j, i, loss.data[0])
#                 loss1.append(loss.data[0])
#                 cosSim = CosSim(chd.data, wdc.data)
#                 eucSim = EucSim(chd.data, wdc.data)
#                 losscos.append(cosSim)
#                 losseuc.append(eucSim)
#                 print(cosSim)
#                 print(eucSim)
#             optimizer.step()
#             wec = wec.data.view(word_dim, -1)
#         
#     plt.plot(range(100), loss1, label="lossa", color="red")
#     plt.plot(range(100), losscos, "b--", label="cos", color="blue")
#     plt.plot(range(100), losseuc, "c--", label="euc", color="green")
#     plt.legend()
#     plt.show()
#     
# 
#     
#     
# 
# def build_matrix(line, vector, vocab):
#     sents = line.split('\t')
#     lsent = sents[1].split()
#     rsent = sents[2].split()
#      
#     label = torch.LongTensor(1, 1)
#     lsentm = torch.FloatTensor(len(lsent), args.word_dim)
#     rsentm = torch.FloatTensor(len(rsent), args.word_dim)
#      
#     label[0] = int(sents[0])
#      
#     for i in range(len(lsent)):
#         if lsent[i] in vocab:
#             lsentm[i] = torch.from_numpy(vector[lsent[i]]).view(1, args.word_dim)
#         else:
#             wv = torch.Tensor(1, args.word_dim)
#             lsentm[i] = init.normal(wv, 0, 0.1)
#              
#     for j in range(len(rsent)):
#         if rsent[j] in vocab:
#             rsentm[j] = torch.from_numpy(vector[rsent[j]]).view(1, args.word_dim)
#         else:
#             wv = torch.Tensor(1, args.word_dim)
#             rsentm[j] = init.normal(wv, 0, 0.1)
#              
#     return label, lsentm, rsentm
#      
# def mainDemo():
#     word_dim = 10
#     word_num = 20
#     hid_size = 10
#     
#     v1 = torch.randn(1,5)
#     v2 = torch.randn(1,5)
#     print(v1)
#     print(norm(v1))
#     print(v2)
#     print(norm(v2))
#     print(cos_sim(v2, v2))
#     print(euc_sim(v1, v2))
#     
#     emb_file = "/Users/hjp/MacBook/Workspace/Workshop/Corpus/bin/text.txt"
#     
#     voc, vec = build_vector(emb_file)
#     #print(voc)
#     print(vec)
#     
    

       


# 
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# 
# torch.manual_seed(123456)
# 
# def euclidean(x,y):
#     sumSq=0.0
#      
#     #add up the squared differences
#     for i in range(len(x)):
#         sumSq+=(x[i]-y[i])**2
#          
#     #take the square root of the result
#     return (sumSq**0.5)
# 
# '''
# class EuclideanDistance(nn.Module):
#     def __init__(self):
#         super(EuclideanDistance, self).__init__()
#         
#     def forward(self, x, y):
#         return ((y-x).mul(y-x)).sqrt().sum()
# '''
# class RecursiveAutoEncoder(nn.Module):
#     def __init__(self, word_dim, hid_dim):
#         super(RecursiveAutoEncoder, self).__init__()
#         self.Wec = nn.Linear(2 * word_dim, word_dim)
#         #self.Weh = nn.Linear(word_dim, hid_dim)
#         #self.Wdh = nn.Linear(hid_dim, word_dim)
#         self.Wde = nn.Linear(word_dim, 2 * word_dim)
#         self.tanh = nn.Tanh()
#     
#     def forward(self, chd):        
#         wec = self.tanh(self.Wec(chd))
#         #weh = self.tanh(self.Weh(wec))
#         #wdh = self.tanh(self.Wdh(weh))
#         wde = self.tanh(self.Wde(wec))
#         
#         return wec, wde
# 
# def main():
#     word_dim = 5
#     hid_dim = 10
#     word_num = 2
# 
#     rae = RecursiveAutoEncoder(word_dim, hid_dim)
#     print(rae)
#     criterion = nn.MSELoss()
#     optimizer = optim.SGD(rae.parameters(), lr=1e-2)
#     
#     input = Variable(torch.randn(word_num, word_dim))
#     rec = Variable(torch.randn(word_num, word_dim).zero_())
#     print(input)
#     print(rec)
#     wec = Variable(torch.randn(1, word_dim).zero_())
#     
#     for i in range(1, word_dim):
#         if i == 1:
#             chd = (torch.cat((input[0], input[1]), 0)).view(1, -1)
#             print(chd)
#             wec, wde = rae(chd)
#             print('wec:')
#             print(wec)
#             print(wde)
#             print('wde:')
#             print(wde.data[0][:word_dim])
#             rec[0] = wde.data[0][:word_dim]
#             rec[1] = wde.data[0][word_dim:]
#             print(wde.data[0][word_dim:])
#             print(rec)
#             print(chd)
#             print(wde)
#             loss = criterion(input, rec)  
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()  
#             '''
#         else:
#             print('wec:')
#             print(wec)
#             print(input[i])
#             chd = (torch.cat((input[i], wec.view(word_dim, -1)), 0)).view(1, -1)
#             wec, wde = rae(chd)
#             print(wec)
#             rec[i] = wde.data[0][:word_dim]
#             print(rec)
#             '''
# '''            
#     loss = criterion(input, rec)
#     print('input:')
#     print(input)
#     print('rec:')
#     print(rec)
#     
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
# '''            
# 
# 
# if __name__ == "__main__":
#     main()

# # -*- coding utf-8 -*-
# """
# @date: Created on Sun Jun 25 06:20:18 2017.
# @description: There is a recursive neural network based on semantic composition for textual similarity.
# @author: Jiangping Huang (hjp@whu.edu.cn).
# @copyright: All right reserved.
# @license: The code is released under the Apache License.
# @version: Alpha.
# """
# 
# import os
# import math
# import time
# import random
# import argparse
# import numpy as np
# 
# import torch
# import torch.nn as nn
# from torch.nn import init
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.autograd import Variable
# 
# parser = argparse.ArgumentParser()
# 
# parser.add_argument('--word_dim', dest='word_dim', type=int, help='embedding dimension', default=20)
# parser.add_argument('--hid_dim', dest='hid_dim', type=int, help='hidden dimension', default=50)
# parser.add_argument('--epochs', dest='epochs', type=int, help='epochs for model training', default=10)
# parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate', default=1e-3)
# parser.add_argument('--num_class', dest='num_class', type=int, help='classes of output', default=2)
# parser.add_argument('--sem_com', dest='sem_com', type=bool, help='semantic composition', default=True)
# parser.add_argument('--seed', dest='seed', type=int, help='random seed', default=1234567890)
# parser.add_argument('--emb_file', dest='emb_file', type=str, help='embedding file path', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/bin/text.txt')
# parser.add_argument('--data_file', dest='data_file', type=str, help='data file path', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/tmp/')
# 
# use_gpu = torch.cuda.is_available()
# 
# args = parser.parse_args()
# torch.manual_seed(args.seed)
# 
# class RvNN(nn.Module):
#     def __init__(self, word_dim, hid_size, num_class):
#         super(RvNN, self).__init__()
#         self.linear11 = nn.Linear(2 * word_dim, word_dim)
#         self.linear12 = nn.Linear(word_dim, word_dim)
#         self.linear21 = nn.Linear(2 * word_dim, word_dim)
#         self.linear22 = nn.Linear(word_dim, word_dim)
#         self.linear3 = nn.Linear(2 * word_dim, hid_size)
#         self.linear4 = nn.Linear(hid_size, hid_size / 10)
#         self.linear5 = nn.Linear(hid_size / 10, num_class)
#         self.relu = nn.ReLU()
#         #self.softmax = nn.LogSoftmax()
#     
#         #self.distance = F.cosine_similarity()
#         
#     def forward(self, left, right, word_dim):
#         c1 = Variable(torch.FloatTensor(torch.zeros(word_dim)))
#         c2 = Variable(torch.FloatTensor(torch.zeros(word_dim)))
#         
#         for i in range(len(left) - 1):
#             if i == 0:
#                 c1 = self.linear11((torch.cat((left[i], left[i + 1]), 0)).view(1, -1))
#                 #c1 = self.relu(self.linear2(c1))
#                 c1 = self.linear12(c1)
#             else:
#                 c1 = self.linear11((torch.cat((c1.view(word_dim, -1), left[i + 1]), 0)).view(1, -1))
#                 #c1 = self.relu(self.linear2(c1))
#                 c1 = self.linear12(c1)
#                 
#         for j in range(len(right) - 1):
#             if j == 0:
#                 c2 = self.linear21((torch.cat((right[j], right[j + 1]), 0)).view(1, -1))
#                 #c2 = self.relu(self.linear2(c2))
#                 c2 = self.linear22(c2)
#             else:
#                 c2 = self.linear21((torch.cat((c2.view(word_dim, -1), right[j + 1]), 0)).view(1, -1))
#                 #c2 = self.relu(self.linear2(c2))
#                 c2 = self.linear22(c2)
#         
#         #print("c1: ", c1)
#         #print("c2: ", c2)        
#         #concat = self.relu(self.linear3(torch.cat((c1, c2), -1)))
#         #print("concat: ", concat)
#         #output = self.softmax(self.linear5(self.relu(self.linear4(concat)))) 
# 
#         c11 = self.relu(c1)
# 
#         c21 = self.relu(c2)
# 
#         output = F.cosine_similarity(c11, c21)
#         return output      
# 
# def set_timer(sec):
#     min = math.floor(sec / 60)
#     sec -= min * 60
#     return '%dm %ds' % (min, sec)
# 
# def build_vector(path):
#     vector = {}
#     vocab = []
# 
#     with open(path, 'r') as lines:
#         for line in lines:
#             tokens = line.split()
#             vocab.append(tokens[0])
#             vector[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
#         lines.close()  
#           
#     return vector, vocab
# 
# def build_corpus(path):
#     trainSet = []
#     testSet = []
#     
#     assert os.path.exists(path)
#     
#     train = os.path.join(path, 'train.txt')
#     test = os.path.join(path, 'test.txt')
#     
#     with open(train, 'r') as lines:
#         for line in lines:
#             sents = line.lower().split('\t')
#             trainSet.append(sents[0] + "\t" + sents[2] + "\t" + sents[6] + "\t" + sents[4] + "\t" + sents[8])
#             
#     with open(test, 'r') as lines:
#         for line in lines:
#             sents = line.lower().split('\t')
#             testSet.append(sents[0] + "\t" + sents[2] + "\t" + sents[6] + "\t" + sents[4] + "\t" + sents[8])
#     
#     return trainSet, testSet
# 
# def build_semcom(line, vector, vocab):
#     sents = line.split('\t')
#     label = torch.LongTensor(1, 1)
#     label[0] = int(sents[0])
#     
#     lsent = sents[1].split()
#     rsent = sents[2].split()
#     ltags = sents[3].split()
#     rtags = sents[4].split()
#     
#     lrow, rrow = 0, 0
#     for i in range(len(ltags)):
#         if ltags[i][0:1] == 'b' or ltags[i][0:1] == 'o':
#             lrow += 1
#     for j in range(len(rtags)):
#         if rtags[j][0:1] == 'b' or rtags[j][0:1] == 'o':
#             rrow += 1
#     lsentm = torch.FloatTensor(lrow, args.word_dim)
#     rsentm = torch.FloatTensor(rrow, args.word_dim)
#     
#     lidx, ridx = 0, 0
#     for i in range(len(ltags)):
#         if ltags[i][0:1] == 'b' or ltags[i][0:1] == 'o':
#             if lsent[i] in vocab:
#                 lsentm[lidx] = torch.from_numpy(vector[lsent[i]]).view(1, args.word_dim)
#             else:
#                 wv = torch.Tensor(1, args.word_dim)
#                 lsentm[lidx] = init.normal(wv, 0, 0.1)
#             if lidx < lrow - 1:
#                 lidx = lidx + 1
#         else:
#             if lsent[i] in vocab:
#                 lsentm[lidx] = lsentm[lidx] + torch.from_numpy(vector[lsent[i]]).view(1, args.word_dim)
#             else:
#                 wv = torch.Tensor(1, args.word_dim)
#                 lsentm[lidx] = lsentm[lidx] + init.normal(wv, 0, 0.1)
#                 
#     for j in range(len(rtags)):
#         if rtags[j][0:1] == 'b' or rtags[j][0:1] == 'o':
#             if rsent[j] in vocab:
#                 rsentm[ridx] = torch.from_numpy(vector[rsent[j]]).view(1, args.word_dim)
#             else:
#                 wv = torch.Tensor(1, args.word_dim)
#                 rsentm[ridx] = init.normal(wv, 0, 0.1)
#             if ridx < rrow - 1:
#                 ridx = ridx + 1
#         else:
#             if rsent[j] in vocab:
#                 rsentm[ridx] = rsentm[ridx] + torch.from_numpy(vector[rsent[j]]).view(1, args.word_dim)
#             else:
#                 wv = torch.Tensor(1, args.word_dim)
#                 rsentm[ridx] = rsentm[ridx] + init.normal(wv, 0, 0.1)
#     
#     return label, lsentm, rsentm    
# 
# def build_matrix(line, vector, vocab):
#     sents = line.split('\t')
#     lsent = sents[1].split()
#     rsent = sents[2].split()
#     
#     label = torch.LongTensor(1, 1)
#     lsentm = torch.FloatTensor(len(lsent), args.word_dim)
#     rsentm = torch.FloatTensor(len(rsent), args.word_dim)
#     
#     label[0] = int(sents[0])
#     
#     for i in range(len(lsent)):
#         if lsent[i] in vocab:
#             lsentm[i] = torch.from_numpy(vector[lsent[i]]).view(1, args.word_dim)
#         else:
#             wv = torch.Tensor(1, args.word_dim)
#             lsentm[i] = init.normal(wv, 0, 0.1)
#             
#     for j in range(len(rsent)):
#         if rsent[j] in vocab:
#             rsentm[j] = torch.from_numpy(vector[rsent[j]]).view(1, args.word_dim)
#         else:
#             wv = torch.Tensor(1, args.word_dim)
#             rsentm[j] = init.normal(wv, 0, 0.1)
#             
#     return label, lsentm, rsentm
# 
# def train(trainSet, vector, vocab, rnn):
#     criterion = nn.L1Loss()
#     optimizer = optim.ASGD(rnn.parameters(), lr=args.learning_rate)
#     
#     for i in range(args.epochs):
#         start = time.time()
#         for j in range(len(trainSet)):
#             if args.sem_com:
#                 label, lsentm, rsentm = build_semcom(trainSet[j], vector, vocab)
#             else:
#                 label, lsentm, rsentm = build_matrix(trainSet[j], vector, vocab)
#                 
#             if use_gpu:
#                 label = Variable(label[0].cuda())
#                 lsentm = Variable(lsentm.cuda())
#                 rsentm = Variable(rsentm.cuda())
#             else:
#                 label = Variable(label[0])
#                 lsentm = Variable(lsentm)
#                 rsentm = Variable(rsentm)
#             
#             optimizer.zero_grad()
#             pred = rnn(rsentm, rsentm, args.word_dim)
# #             print("The epoch %d " % (i+1) + "and %d example." % (j+1))
#             print(pred)
# #             print('label')
# #             print(label)
#             
#             loss = criterion(pred, label)
#             print(loss.data)
#             loss.backward()
#             optimizer.step()
#             
#         end = time.time()
#         print("The epoch %d cost " % (i + 1) + set_timer(end - start))    
#     
# def test(testSet, vector, vocab):
#     for i in range(len(testSet)):
#         print(testSet[i])        
# 
# if __name__ == "__main__":
#     print(args)
#     start = time.time()
#     
#     vector, vocab = build_vector(args.emb_file)
#     trainSet, testSet = build_corpus(args.data_file)
#     
#     random.shuffle(trainSet)
#     random.shuffle(testSet)
#     
#     rnn = RvNN(args.word_dim, args.hid_dim, args.num_class)
#     print(rnn)    
#     
#     if use_gpu:
#         rnn = rnn.cuda()
#         
#     train(trainSet, vector, vocab, rnn)
#     
#     end = time.time()
#     print("The model cost " + set_timer(end - start) + " for training.")


'''rae'''
# '''
# Created on Jul 15, 2017
# 
# @author: hjp
# '''
# 
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# 
# torch.manual_seed(123456)
# 
# def euclidean(x,y):
#     sumSq=0.0
#      
#     #add up the squared differences
#     for i in range(len(x)):
#         sumSq+=(x[i]-y[i])**2
#          
#     #take the square root of the result
#     return (sumSq**0.5)
# 
# '''
# class EuclideanDistance(nn.Module):
#     def __init__(self):
#         super(EuclideanDistance, self).__init__()
#         
#     def forward(self, x, y):
#         return ((y-x).mul(y-x)).sqrt().sum()
# '''
# class RecursiveAutoEncoder(nn.Module):
#     def __init__(self, word_dim, hid_dim):
#         super(RecursiveAutoEncoder, self).__init__()
#         self.Wec = nn.Linear(2 * word_dim, word_dim)
#         #self.Weh = nn.Linear(word_dim, hid_dim)
#         #self.Wdh = nn.Linear(hid_dim, word_dim)
#         self.Wde = nn.Linear(word_dim, 2 * word_dim)
#         self.tanh = nn.Tanh()
#     
#     def forward(self, chd):        
#         wec = self.tanh(self.Wec(chd))
#         #weh = self.tanh(self.Weh(wec))
#         #wdh = self.tanh(self.Wdh(weh))
#         wde = self.tanh(self.Wde(wec))
#         
#         return wec, wde
# 
# def main():
#     word_dim = 5
#     hid_dim = 10
#     word_num = 2
# 
#     rae = RecursiveAutoEncoder(word_dim, hid_dim)
#     print(rae)
#     criterion = nn.MSELoss()
#     optimizer = optim.SGD(rae.parameters(), lr=1e-2)
#     
#     input = Variable(torch.randn(word_num, word_dim))
#     rec = Variable(torch.randn(word_num, word_dim).zero_())
#     print(input)
#     print(rec)
#     wec = Variable(torch.randn(1, word_dim).zero_())
#     
#     for i in range(1, word_dim):
#         if i == 1:
#             chd = (torch.cat((input[0], input[1]), 0)).view(1, -1)
#             print(chd)
#             wec, wde = rae(chd)
#             print('wec:')
#             print(wec)
#             print(wde)
#             print('wde:')
#             print(wde.data[0][:word_dim])
#             rec[0] = wde.data[0][:word_dim]
#             rec[1] = wde.data[0][word_dim:]
#             print(wde.data[0][word_dim:])
#             print(rec)
#             print(chd)
#             print(wde)
#             loss = criterion(input, rec)  
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()  
#             '''
#         else:
#             print('wec:')
#             print(wec)
#             print(input[i])
#             chd = (torch.cat((input[i], wec.view(word_dim, -1)), 0)).view(1, -1)
#             wec, wde = rae(chd)
#             print(wec)
#             rec[i] = wde.data[0][:word_dim]
#             print(rec)
#             '''
# '''            
#     loss = criterion(input, rec)
#     print('input:')
#     print(input)
#     print('rec:')
#     print(rec)
#     
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
# '''            
# 
# 
# if __name__ == "__main__":
#     main()



'''
test
'''

# from numpy import dot
# from numpy.linalg import norm
# 
# import math
# 
# import torch
# import torch.nn as nn
# import torch.autograd as autograd
# from torch.autograd import Variable
# from torch import optim
# 
# torch.manual_seed(2718281828459045235360)
# 
# 
# class RAE(nn.Module):
#     def __init__(self, word_dim):
#         super(RAE, self).__init__()
#         self.We = nn.Linear(2*word_dim, word_dim)
#         self.Weh = nn.Linear(word_dim, 100*word_dim)
#         self.Wdh = nn.Linear(100*word_dim, word_dim)
#         self.Wd = nn.Linear(word_dim, 2 * word_dim)
#         self.tanh = nn.Tanh()
#         
#     def forward(self, chd):
#         enc = self.tanh(self.We(chd))
#         dec = self.tanh(self.Wd(enc))
#         
#         return enc, dec
#     
# 
# # def cosine_similarity(v1,v2):
# #     "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
# #     sumxx, sumxy, sumyy = 0, 0, 0
# #     for i in range(len(v1)):
# #         x = v1[i]; y = v2[i]
# #         sumxx += x*x
# #         sumyy += y*y
# #         sumxy += x*y
# #     return sumxy/math.sqrt(sumxx*sumyy)
# # 
# # v1,v2 = [3, 45, 7, 2], [2, 54, 13, 15]
# # print(v1, v2, cosine_similarity(v1,v2))
#     
# word_dim = 5
# epochs = 3000
# word_num = 5
# input = torch.randn(word_num, word_dim)
# rae = RAE(word_dim)
# print(rae)
# print(input)
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(rae.parameters(), lr=1e-2)
# 
# for i in range(1, word_num):
#     if i == 1:
#         chd = Variable((torch.cat((input[0], input[1]), 0)).view(1, -1))
#         enc, dec = rae(chd)
#     
#     
#     #print(chd)
# 
#     
# 
# 
# 
#     loss = criterion(dec, chd)
# 
# 
#     print(i, loss.data[0])
# 
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     
#     print(enc)
#     print(dec)
#     print(input[2])
#     print(enc.view(word_dim, -1))
#     chd = Variable((torch.cat((input[2], enc.data.view(word_dim, -1)), 0)).view(1, -1))
#     print(chd)
#     
#     enc, dec = rae(chd)
#     loss = criterion(dec, chd)
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(dec)
#     print(chd)
#     
#     
#     
# #     if loss.data[0] < 0.055:
# #         break
#     



'''demo'''
# # -*- coding utf-8 -*-
# """
# @date: Created on Sun Jun 25 06:20:18 2017.
# @description: There is a recursive neural network based on semantic composition for textual similarity.
# @author: Jiangping Huang (hjp@whu.edu.cn).
# @copyright: All right reserved.
# @license: The code is released under the Apache License.
# @version: Alpha.
# """
# 
# import os
# import math
# import time
# import random
# import argparse
# import numpy as np
# 
# import torch
# import torch.nn as nn
# from torch.nn import init
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.autograd import Variable
# 
# parser = argparse.ArgumentParser()
# 
# parser.add_argument('--word_dim', dest='word_dim', type=int, help='embedding dimension', default=20)
# parser.add_argument('--hid_dim', dest='hid_dim', type=int, help='hidden dimension', default=50)
# parser.add_argument('--epochs', dest='epochs', type=int, help='epochs for model training', default=10)
# parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate', default=1e-3)
# parser.add_argument('--num_class', dest='num_class', type=int, help='classes of output', default=2)
# parser.add_argument('--sem_com', dest='sem_com', type=bool, help='semantic composition', default=True)
# parser.add_argument('--seed', dest='seed', type=int, help='random seed', default=1234567890)
# parser.add_argument('--emb_file', dest='emb_file', type=str, help='embedding file path', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/bin/text.txt')
# parser.add_argument('--data_file', dest='data_file', type=str, help='data file path', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/tmp/')
# 
# use_gpu = torch.cuda.is_available()
# 
# args = parser.parse_args()
# torch.manual_seed(args.seed)
# 
# class RAE(nn.Module):
#     def __init__(self, word_dim, hid_size):
#         super(RAE, self).__init__()
#         self.linear1 = nn.Linear(2 * word_dim, word_dim)
#         self.linear2 = nn.Linear(word_dim, 2 * word_dim)
#         self.relu = nn.Tanh()
#         self.softmax = nn.LogSoftmax()
#         
#     def forward(self, left, word_dim):
#         c1 = Variable(torch.FloatTensor(torch.zeros(word_dim)))
#         c2 = Variable(torch.FloatTensor(torch.zeros(word_dim)))
#         
#         for i in range(len(left) - 1):
#             if i == 0:
#                 c1 = self.relu(self.linear1((torch.cat((left[i], left[i + 1]), 0)).view(1, -1)))
#                 print c1
#                 c12 = self.relu(self.linear2(c1))
#                 print c12
#             else:
#                 c1 = self.relu(self.linear1((torch.cat((c1.view(word_dim, -1), left[i + 1]), 0)).view(1, -1)))
#                 print c1
#                 c12 = self.relu(self.linear2(c1))
#                 print c12
#                 
#         return c1, c12
#                 
#                 
#                 
# WD = 10
# HD = 20
# 
# rae = RAE(10, 20)
# 
# input = torch.randn(5, 10)
# print input
# output = rae.forward(Variable(input), WD)
# print output
#    
# """
# def set_timer(sec):
#     min = math.floor(sec / 60)
#     sec -= min * 60
#     return '%dm %ds' % (min, sec)
# 
# def build_vector(path):
#     vector = {}
#     vocab = []
# 
#     with open(path, 'r') as lines:
#         for line in lines:
#             tokens = line.split()
#             vocab.append(tokens[0])
#             vector[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
#         lines.close()  
#           
#     return vector, vocab
# 
# def build_corpus(path):
#     trainSet = []
#     testSet = []
#     
#     assert os.path.exists(path)
#     
#     train = os.path.join(path, 'train.txt')
#     test = os.path.join(path, 'test.txt')
#     
#     with open(train, 'r') as lines:
#         for line in lines:
#             sents = line.lower().split('\t')
#             trainSet.append(sents[0] + "\t" + sents[2] + "\t" + sents[6] + "\t" + sents[4] + "\t" + sents[8])
#             
#     with open(test, 'r') as lines:
#         for line in lines:
#             sents = line.lower().split('\t')
#             testSet.append(sents[0] + "\t" + sents[2] + "\t" + sents[6] + "\t" + sents[4] + "\t" + sents[8])
#     
#     return trainSet, testSet
# 
# def build_semcom(line, vector, vocab):
#     sents = line.split('\t')
#     label = torch.LongTensor(1, 1)
#     label[0] = int(sents[0])
#     
#     lsent = sents[1].split()
#     rsent = sents[2].split()
#     ltags = sents[3].split()
#     rtags = sents[4].split()
#     
#     lrow, rrow = 0, 0
#     for i in range(len(ltags)):
#         if ltags[i][0:1] == 'b' or ltags[i][0:1] == 'o':
#             lrow += 1
#     for j in range(len(rtags)):
#         if rtags[j][0:1] == 'b' or rtags[j][0:1] == 'o':
#             rrow += 1
#     lsentm = torch.FloatTensor(lrow, args.word_dim)
#     rsentm = torch.FloatTensor(rrow, args.word_dim)
#     
#     lidx, ridx = 0, 0
#     for i in range(len(ltags)):
#         if ltags[i][0:1] == 'b' or ltags[i][0:1] == 'o':
#             if lsent[i] in vocab:
#                 lsentm[lidx] = torch.from_numpy(vector[lsent[i]]).view(1, args.word_dim)
#             else:
#                 wv = torch.Tensor(1, args.word_dim)
#                 lsentm[lidx] = init.normal(wv, 0, 0.1)
#             if lidx < lrow - 1:
#                 lidx = lidx + 1
#         else:
#             if lsent[i] in vocab:
#                 lsentm[lidx] = lsentm[lidx] + torch.from_numpy(vector[lsent[i]]).view(1, args.word_dim)
#             else:
#                 wv = torch.Tensor(1, args.word_dim)
#                 lsentm[lidx] = lsentm[lidx] + init.normal(wv, 0, 0.1)
#                 
#     for j in range(len(rtags)):
#         if rtags[j][0:1] == 'b' or rtags[j][0:1] == 'o':
#             if rsent[j] in vocab:
#                 rsentm[ridx] = torch.from_numpy(vector[rsent[j]]).view(1, args.word_dim)
#             else:
#                 wv = torch.Tensor(1, args.word_dim)
#                 rsentm[ridx] = init.normal(wv, 0, 0.1)
#             if ridx < rrow - 1:
#                 ridx = ridx + 1
#         else:
#             if rsent[j] in vocab:
#                 rsentm[ridx] = rsentm[ridx] + torch.from_numpy(vector[rsent[j]]).view(1, args.word_dim)
#             else:
#                 wv = torch.Tensor(1, args.word_dim)
#                 rsentm[ridx] = rsentm[ridx] + init.normal(wv, 0, 0.1)
#     
#     return label, lsentm, rsentm    
# 
# def build_matrix(line, vector, vocab):
#     sents = line.split('\t')
#     lsent = sents[1].split()
#     rsent = sents[2].split()
#     
#     label = torch.LongTensor(1, 1)
#     lsentm = torch.FloatTensor(len(lsent), args.word_dim)
#     rsentm = torch.FloatTensor(len(rsent), args.word_dim)
#     
#     label[0] = int(sents[0])
#     
#     for i in range(len(lsent)):
#         if lsent[i] in vocab:
#             lsentm[i] = torch.from_numpy(vector[lsent[i]]).view(1, args.word_dim)
#         else:
#             wv = torch.Tensor(1, args.word_dim)
#             lsentm[i] = init.normal(wv, 0, 0.1)
#             
#     for j in range(len(rsent)):
#         if rsent[j] in vocab:
#             rsentm[j] = torch.from_numpy(vector[rsent[j]]).view(1, args.word_dim)
#         else:
#             wv = torch.Tensor(1, args.word_dim)
#             rsentm[j] = init.normal(wv, 0, 0.1)
#             
#     return label, lsentm, rsentm
# 
# def train(trainSet, vector, vocab, rnn):
#     criterion = nn.NLLLoss()
#     optimizer = optim.ASGD(rnn.parameters(), lr=args.learning_rate)
#     
#     for i in range(args.epochs):
#         start = time.time()
#         for j in range(len(trainSet)):
#             if args.sem_com:
#                 label, lsentm, rsentm = build_semcom(trainSet[j], vector, vocab)
#             else:
#                 label, lsentm, rsentm = build_matrix(trainSet[j], vector, vocab)
#                 
#             if use_gpu:
#                 label = Variable(label[0].cuda())
#                 lsentm = Variable(lsentm.cuda())
#                 rsentm = Variable(rsentm.cuda())
#             else:
#                 label = Variable(label[0])
#                 lsentm = Variable(lsentm)
#                 rsentm = Variable(rsentm)
#             
#             optimizer.zero_grad()
#             pred = rnn(rsentm, rsentm, args.word_dim)
#             print("The epoch %d " % (i+1) + "and %d example." % (j+1))
#             print(pred.data)
#             
#             loss = criterion(pred, label)
#             print(loss.data)
#             loss.backward()
#             optimizer.step()
#             
#         end = time.time()
#         print("The epoch %d cost " % (i + 1) + set_timer(end - start))    
#     
# def test(testSet, vector, vocab):
#     for i in range(len(testSet)):
#         print(testSet[i])        
# 
# if __name__ == "__main__":
#     print(args)
#     start = time.time()
#     
#     vector, vocab = build_vector(args.emb_file)
#     trainSet, testSet = build_corpus(args.data_file)
#     
#     random.shuffle(trainSet)
#     random.shuffle(testSet)
#     
#     rnn = RvNN(args.word_dim, args.hid_dim, args.num_class)
#     print(rnn)    
#     
#     if use_gpu:
#         rnn = rnn.cuda()
#         
#     train(trainSet, vector, vocab, rnn)
#     
#     end = time.time()
#     print("The model cost " + set_timer(end - start) + " for training.")
# 
# """



'''
#"demo code for loss function"
'''
# import math
# import argparse
# import matplotlib.pyplot as plt
# 
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# 
# parser = argparse.ArgumentParser()
# 
# parser.add_argument('--word_dim', dest='word_dim', type=int, help='word embedding dimension', default=20)
# parser.add_argument('--hid_size', dest='hid_size', type=int, help='hidden dimension size', default=150)
# parser.add_argument('--enc_size', dest='enc_size', type=int, help='encode dimension size', default=50)
# parser.add_argument('--epochs', dest='epochs', type=int, help='epochs for model training', default=1000)
# parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate', default=5e-3)
# parser.add_argument('--num_cate', dest='num_cate', type=int, help='categories for output', default=5)
# parser.add_argument('--alpha', dest='alpha', type=float, help='ratio in different criterion', default=0.3)
# parser.add_argument('--init_weight', dest='init_weight', type=float, help='initial weight', default=0.1)
# parser.add_argument('--sem_com', dest='sem_com', type=bool, help='if semantic composition', default=False)
# parser.add_argument('--seed', dest='seed', type=long, help='random seed', default=2718281828459045232536)
# parser.add_argument('--emb_file', dest='emb_file', type=str, help='embedding file', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/bin/text.txt')
# parser.add_argument('--data_file', dest='data_file', type=str, help='data file', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/ssc/')
# 
# args = parser.parse_args()
# use_gpu = torch.cuda.is_available()
# print(args)
# torch.manual_seed(args.seed)
# 
# class RecursiveAutoEncoder(nn.Module):
#     def __init__(self, word_dim, hid_size, enc_size, num_cate):
#         super(RecursiveAutoEncoder, self).__init__()
#         self.WeC2P = nn.Linear(2*word_dim, word_dim)
#         self.WeP2H = nn.Linear(word_dim, hid_size)
#         self.WeH2E = nn.Linear(hid_size, enc_size)
#         self.WeE2C = nn.Linear(enc_size, num_cate)
#         self.WdE2H = nn.Linear(enc_size, hid_size)
#         self.WdH2P = nn.Linear(hid_size, word_dim)
#         self.WdP2C = nn.Linear(word_dim, 2*word_dim)
#         self.tanh = nn.Tanh()
#         self.relu = nn.ReLU()
#         self.soft = nn.LogSoftmax()
#         
#     def encoder(self, children):
#         parent = self.tanh(self.WeC2P(children))
#         encode = self.tanh(self.WeH2E(self.tanh(self.WeP2H(parent))))
#         scores = self.soft(self.tanh(self.WeE2C(encode)))
#         return parent, encode, scores
#     
#     def decoder(self, encode):
#         return self.tanh(self.WdP2C(self.tanh(self.WdH2P(self.tanh(self.WdE2H(encode))))))
#     
#     def forward(self, children):
#         parent, encode, scores = self.encoder(children)
#         decode = self.decoder(encode)
#         return parent, encode, scores, decode
#     
# rae = RecursiveAutoEncoder(args.word_dim, args.hid_size, args.enc_size, args.num_cate)
# print(rae)
# criterion_mse = nn.MSELoss(size_average=True)
# optimizer = optim.SGD(rae.parameters(), lr=args.learning_rate)
# criterion_nll = nn.NLLLoss()
# 
# sentm = Variable(torch.FloatTensor(torch.randn(10, 20, 20)))
# label = Variable(torch.LongTensor([1, 2, 1, 4, 0, 2, 3, 0, 2, 3]))
# loss_storage = []
# 
# for k in range(args.epochs):
#     for i in range(len(sentm)):
#         parent = sentm[i][0]
#         for j in range(1, len(sentm[i])):
#             children = (torch.cat((sentm[i][j], parent), 0)).view(1, -1)
#             parent, encode, scores, decode = rae(children)
#             parent = Variable(parent.data.view(args.word_dim, -1))
#             if j < len(sentm[i]) - 1:
#                 optimizer.zero_grad()
#                 loss = criterion_mse(decode, children)
#                 loss.backward()
#                 optimizer.step()
#             else:
#                 optimizer.zero_grad()
#                 loss = args.alpha * criterion_mse(decode, children) + (1 - args.alpha) * criterion_nll(scores, label[i])
#                 if i == 5:
#                     print(k, i, loss.data[0])
#                     loss_storage.append(loss.data[0])
#                 loss.backward()
#                 optimizer.step()
# 
# plt.plot(range(args.epochs), loss_storage, label="loss_valid", color="red")
# plt.plot(range(args.epochs). accu_storage, label="accu_valid", color="green")
# plt.legend()
# plt.show()
#             
#         
#         

#multi-label examples
# import torch
# import torch.nn as nn
# import numpy as np
# import torch.optim as optim
# from torch.autograd import Variable
# 
# # (1, 0) => target labels 0+2
# # (0, 1) => target labels 1
# # (1, 1) => target labels 3
# train = []
# labels = []
# for i in range(10000):
#     category = (np.random.choice([0, 1]), np.random.choice([0, 1]))
#     if category == (1, 0):
#         train.append([np.random.uniform(0.1, 1), 0])
#         labels.append([1, 0, 1])
#     if category == (0, 1):
#         train.append([0, np.random.uniform(0.1, 1)])
#         labels.append([0, 1, 0])
#     if category == (0, 0):
#         train.append([np.random.uniform(0.1, 1), np.random.uniform(0.1, 1)])
#         labels.append([0, 0, 1])
# 
# class _classifier(nn.Module):
#     def __init__(self, nlabel):
#         super(_classifier, self).__init__()
#         self.main = nn.Sequential(
#             nn.Linear(2, 64),
#             nn.ReLU(),
#             nn.Linear(64, nlabel),
#         )
# 
#     def forward(self, input):
#         return self.main(input)
# 
# nlabel = len(labels[0]) # => 3
# classifier = _classifier(nlabel)
# 
# optimizer = optim.Adam(classifier.parameters())
# criterion = nn.MultiLabelSoftMarginLoss()
# 
# epochs = 5
# for epoch in range(epochs):
#     losses = []
#     for i, sample in enumerate(train):
#         inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
#         labelsv = Variable(torch.FloatTensor(labels[i])).view(1, -1)
#         print('flag: ')
#         print(inputv)
#         print(labelsv)
#         output = classifier(inputv)
#         loss = criterion(output, labelsv)
# 
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.data.mean())
#     print('[%d/%d] Loss: %.3f' % (epoch+1, epochs, np.mean(losses)))



#loss example#
# import math
# 
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# 
# 
# class MyCustomLoss():
# 
#     def forward(self, x, y):
#         xx, yy, xy = 0, 0, 0
#         for i in range(len(x[0])):
#             xx = xx + x[0][i] ** 2
#             yy = yy + y[0][i] ** 2
#             xy = xy + x[0][i] * y[0][i]
#         loss = xy / math.sqrt(xx * yy)
#         #... # implementation
#         return loss   # a single number (averaged loss over batch samples)
# 
#     #def backward(self, grad_output):
#        # ... # implementation
#     #   return grad_input, None
# 
# def cos_sim(x, y):
#     xx, yy, xy = 0, 0, 0
#     for i in range(len(x[0])):
#         xx = xx + x[0][i] ** 2
#         yy = yy + y[0][i] ** 2
#         xy = xy + x[0][i] * y[0][i]
#     return xy / math.sqrt(xx * yy)
# 
#    
# inp = Variable(torch.randn(1,10).double(), requires_grad=True)
# print(inp.data)
# target = Variable(torch.randn(1, 10), requires_grad=False)
# loss = MyCustomLoss()(inp.data, target.data)
# loss.backward()


