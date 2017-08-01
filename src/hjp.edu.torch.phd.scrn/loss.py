import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class MyCustomLoss():

    def forward(self, x, y):
        xx, yy, xy = 0, 0, 0
        for i in range(len(x[0])):
            xx = xx + x[0][i] ** 2
            yy = yy + y[0][i] ** 2
            xy = xy + x[0][i] * y[0][i]
        loss = xy / math.sqrt(xx * yy)
        #... # implementation
        return loss   # a single number (averaged loss over batch samples)

    #def backward(self, grad_output):
       # ... # implementation
    #   return grad_input, None

def cos_sim(x, y):
    xx, yy, xy = 0, 0, 0
    for i in range(len(x[0])):
        xx = xx + x[0][i] ** 2
        yy = yy + y[0][i] ** 2
        xy = xy + x[0][i] * y[0][i]
    return xy / math.sqrt(xx * yy)

   
inp = Variable(torch.randn(1,10).double(), requires_grad=True)
print(inp.data)
target = Variable(torch.randn(1, 10), requires_grad=False)
loss = MyCustomLoss()(inp.data, target.data)
loss.backward()