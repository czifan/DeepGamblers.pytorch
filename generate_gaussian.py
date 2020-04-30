# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

def gaussian(mean, std, size):
    d = np.random.normal(0, std, size=size)
    data = np.array([[np.random.choice(d, 1), np.random.choice(d, 1)] for _ in range(size)])
    data[:, 0] += mean[0]
    data[:, 1] += mean[1]
    return data

if __name__ == '__main__':
    data1 = gaussian(mean=(1,1), std=1, size=200)
    data2 = gaussian(mean=(-1,-1), std=1, size=200)
    plt.scatter(data1[:,0], data1[:,1], color='r')
    plt.scatter(data2[:,0], data2[:,1], color='b')
    plt.show()