import os
import csv
import json
import random
import pickle
import torch
import numpy as np
import pandas as pd
import datetime as dt
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim
import matplotlib.pyplot as plt


from config import *
from utils import *

random.seed(12)
np.random.seed(12)
torch.manual_seed(12)
torch.cuda.manual_seed(12)

def print_loss_weight():
    loss_array = []
    for i in range(int(len(alldates)*0.8)):
        loss_array.append(1)
    for i in range(int(len(alldates)*0.8), int(len(alldates))):
        loss_array.append(0)

    random.shuffle(loss_array)
    print(loss_array)


def max_min_mobility(idx):
    csvFile = open('../../row_data/mobility_truth.csv', 'r')
    reader = csv.reader(csvFile)
    begin = later_n_date(alldates[0], -7)
    end = alldates[-1]

    data = []

    for item in reader:
        if reader.line_num == 1:
            continue
        if begin <= item[4] <= end:
            data.append(float(item[5:][idx]))
    
    csvFile.close()
        
    return np.min(data), np.max(data)


def max_min_ump():
    csvFile = open('../../row_data/unemploy.csv', 'r')
    reader = csv.reader(csvFile)
    begin = later_n_date(alldates[0], -7)
    end = alldates[-1]

    data = []

    for item in reader:
        if begin <= item[0] <= end:
            data.append(float(item[1]))
    
    csvFile.close()
    # print(data)
    return np.min(data), np.max(data)


def max_min_inf():
    csvFile = open('../../row_data/E.csv', 'r')
    reader = csv.reader(csvFile)
    begin = alldates[0]
    end = alldates[-1]

    data = []

    for item in reader:
        if begin <= item[0] <= end:
            data.append(float(item[2]))
    
    csvFile.close()
    # print(data)
    return np.min(data), np.max(data)


def max_min_new():
    csvFile = open('../../row_data/confirm.csv', 'r')
    reader = csv.reader(csvFile)
    begin = alldates[0]
    end = alldates[-1]

    data = []
    all_data = []
    for item in reader:
        all_data.append(item)
    
    csvFile.close()

    for i in range(len(all_data)):
        if begin <= all_data[i][0] <= end:
            data.append(float(all_data[i][1])-float(all_data[i-1][1]))

    return np.min(data), np.max(data)


def get_max_min():
    data = {}
    for i in range(6):
        _max, _min = max_min_mobility(i)
        data['m{}'.format(i)] = [_max, _min]
    
    _max, _min = max_min_ump()
    data['ump'] = [_max, _min]
    _max, _min = max_min_inf()
    data['inf'] = [0, _min]
    _max, _min = max_min_new()
    data['new'] = [_max, _min]

    return data

if __name__ == '__main__':
    print_loss_weight()
    # data = {}
    # for i in range(26):
    #     alldates = dataset[i*7:57+i*7]
    #     data[i] = get_max_min()

    # with open('../../row_data/max_min_num.json', 'w') as f:
    #     json.dump(data, f)