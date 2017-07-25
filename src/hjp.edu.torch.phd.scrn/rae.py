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