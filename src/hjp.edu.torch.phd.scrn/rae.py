# '''
# Created on Jul 15, 2017
# 
# @author: hjp
# '''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(314159265)

class RAE(nn.Module):
    def __init__(self, word_dim, hid_dim):
        super(RAE, self).__init__()
        self.Wec = nn.Linear(2 * word_dim, word_dim)
        self.Weh = nn.Linear(word_dim, hid_dim)
        self.Whd = nn.Linear(hid_dim, word_dim)
        self.Wdc = nn.Linear(word_dim, 2 * word_dim)
        self.tanh = nn.Tanh()
        
    def encoder(self):
        x = x


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