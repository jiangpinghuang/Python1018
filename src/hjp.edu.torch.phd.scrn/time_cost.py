#!/usr/bin/env python
#coding=utf-8

import os
import math
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# accu_storage = [1.65, 3.14, 4.98, 6.74, 8.35, 10.01, 11.98, 14.05, 16.19, 18.45]
# loss_storage = [0.86, 1.96, 3.46, 4.94, 6.47, 7.9, 9.3, 10.5, 11.9, 13.27]

accu_storage = [0.95, 2.14, 3.08, 4.04, 5.15, 6.21, 7.48, 8.55, 9.69, 10.65]
loss_storage = [0.66, 1.16, 2.04, 2.74, 3.47, 4.19, 5.03, 5.85, 6.79, 7.60]
# for i in range(10):
#     loss_storage.append(0.1 * i)
    #accu_storage.append(0.1 * i * 2)

plt.plot(np.arange(10, 110, 10), accu_storage, label=u'未采用语义组合方法', color="blue",linestyle="-")     
plt.plot(np.arange(10, 110, 10), loss_storage, label=u'采用语义组合方法', color="green",linestyle=":")
plt.xlabel(u'训练语料比例(%)')
plt.ylabel(u'训练时间(小时)')
my_x_ticks = np.arange(10, 110, 10)
my_y_ticks = np.arange(0, 13, 2)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)

plt.legend()
plt.show()     
          