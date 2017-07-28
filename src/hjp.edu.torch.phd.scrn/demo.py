import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument('--word_dim', dest='word_dim', type=int, help='word embedding dimension', default=20)
parser.add_argument('--hid_size', dest='hid_size', type=int, help='hidden dimension size', default=150)
parser.add_argument('--enc_size', dest='enc_size', type=int, help='encode dimension size', default=50)
parser.add_argument('--epochs', dest='epochs', type=int, help='epochs for model training', default=20)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate', default=1e-2)
parser.add_argument('--num_cate', dest='num_cate', type=int, help='categories for output', default=5)
parser.add_argument('--init_weight', dest='init_weight', type=float, help='initial weight', default=0.1)
parser.add_argument('--sem_com', dest='sem_com', type=bool, help='if semantic composition', default=False)
parser.add_argument('--seed', dest='seed', type=long, help='random seed', default=2718281828459045232536)
parser.add_argument('--emb_file', dest='emb_file', type=str, help='embedding file', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/bin/text.txt')
parser.add_argument('--data_file', dest='data_file', type=str, help='data file', default='/Users/hjp/MacBook/Workspace/Workshop/Corpus/ssc/')

args = parser.parse_args()
use_gpu = torch.cuda.is_available()
print(args)
torch.manual_seed(args.seed)

class RecursiveAutoEncoder(nn.Module):
    def __init__(self, word_dim, hid_size, enc_size, num_cate):
        super(RecursiveAutoEncoder, self).__init__()
        self.WeC2P = nn.Linear(2*word_dim, word_dim)
        self.WeP2H = nn.Linear(word_dim, hid_size)
        self.WeH2E = nn.Linear(hid_size, enc_size)
        self.WeE2C = nn.Linear(enc_size, num_cate)
        self.WdE2H = nn.Linear(enc_size, hid_size)
        self.WdH2P = nn.Linear(hid_size, word_dim)
        self.WdP2C = nn.Linear(word_dim, 2*word_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.soft = nn.Softmax()
        
    def encoder(self, children):
        parent = self.tanh(self.WeC2P(children))
        encode = self.tanh(self.WeH2E(self.tanh(self.WeP2H(self.tanh(self.WeC2P(parent))))))
        scores = self.soft(self.tanh(self.WeE2C(encode)))
        return parent, encode, scores
    
    def decoder(self, encode):
        return self.tanh(self.WdP2C(self.tanh(self.WdH2P(self.tanh(self.WdE2H(encode))))))
    
    def forward(self, children):
        parent, encode, scores = self.encoder(children)
        decode = self.decoder(encode)
        return parent, encode, scores, decode
    
rae = RecursiveAutoEncoder(args.word_dim, args.hid_size, args.enc_size, args.num_cate)
print(rae)
criterion = nn.MSELoss()
optimizer = optim.SGD(rae.parameters(), lr=args.learning_rate)

input = Variable(torch.FloatTensor(torch.randn(10,20, 20)))
print(input)
label = Variable(torch.IntTensor([1, 2, 1, 4, 0, 2, 3, 0, 2, 3]))
print(label)
        
        