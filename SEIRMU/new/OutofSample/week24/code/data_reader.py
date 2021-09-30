import os
import csv
import random
import pickle
import torch
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import *
from utils import *


def read_control():
    with open('../../row_data/controls_max_min.pkl', 'rb') as f:
        all_controls = pickle.load(f)
    control = all_controls['United States']
    control = torch.FloatTensor(control).cuda().view(1, -1)
    return control


def read_confirm_data():
    csvFile = open('../../row_data/confirm.csv', 'r')
    reader = csv.reader(csvFile)

    confirm, remove = [], []

    for item in reader:
        if item[0] in alldates:
            confirm.append(int(item[1]))
            remove.append(int(item[2]))
    
    return confirm, remove


def get_E0():
    csvFile = open('../../row_data/E.csv', 'r')
    reader = csv.reader(csvFile)
    _E0 = 0
    for item in reader:
        if item[0] <= alldates[1]:
            _E0 += float(item[1])
    csvFile.close()

    return _E0

def read_mobility_truth(mode):
    csvFile = open('../../row_data/mobility_truth.csv', 'r')
    reader = csv.reader(csvFile)

    begin_date = alldates[0]
    if mode == 'training':
        begin_date = later_n_date(alldates[0], -6)

    data = []
    for item in reader:
        if reader.line_num == 1:
            continue

        if item[2] != 'United States':
            continue
        if begin_date <= item[4] <= alldates[-1]:
            for i in range(5, 11):
                item[i] = (float(item[i]) - max_min_param['m{}'.format(i-5)][0]) / (max_min_param['m{}'.format(i-5)][1] - max_min_param['m{}'.format(i-5)][0])
            data.append(item[5:])

    csvFile.close()
    return np.array(data)


def read_ump_truth():
    csvFile = open('../../row_data/unemploy.csv', 'r')
    reader = csv.reader(csvFile)

    begin_date = alldates[0]

    data, last = [], []

    state_ump = {}
    for item in reader:
        state_ump[item[0]] = float(item[1])
        state_ump[item[0]] = (state_ump[item[0]] - max_min_param['ump'][0]) / (max_min_param['ump'][1] - max_min_param['ump'][0])
    csvFile.close()

    for d in alldates:
        if d in state_ump and begin_date <= d <= alldates[-1]:
            data.append(state_ump[d])

            last_week = later_n_date(d, -7)
            last.append(state_ump[last_week])
    return data, last


def read_oxc_onehot(end_date=alldates[-1]):
    # read blm data
    csvFile = open('../../row_data/US_BLM_data_7ave_pop.csv', 'r')
    reader = csv.reader(csvFile)

    blm_index = []
    for item in reader:
        if alldates[0] <= item[0] <= end_date:
            item[1] = float(item[1])
            blm_index.append(item[1])
    
    csvFile.close()
    
    # read oxc index
    csvFile = open('../../row_data/oxc_onehot_ave7_index.csv', 'r')
    reader = csv.reader(csvFile)

    data = []
    for item in reader:
        if reader.line_num == 1:
            continue
        
        if alldates[0] <= item[1] <= end_date:
            for i in range(2, len(item)):
                item[i] = float(item[i])
            data.append(item[2:])
    
    csvFile.close()

    for i in range(len(data)):
        data[i] = [blm_index[i]] + data[i]
    return data


def read_infect_rate_truth():
    csvFile = open('../../row_data/E.csv', 'r')
    reader = csv.reader(csvFile)

    data = []
    for item in reader:
        if item[0] in alldates:
            item[2] = (float(item[2]) - max_min_param['inf'][0]) / (max_min_param['inf'][1] - max_min_param['inf'][0])
            data.append(item[2])
    csvFile.close()

    return data

# ------------------------read data pred-----------------------------------------------------------------------------
def read_confirm_pred(mode='self', if_norm='train'):
    begin = later_n_date(alldates[0], -7)
    end = alldates[-1]
    
    if mode == 'self':
        csvFile = open('../snap_data/confirm_pred.csv', 'r')
    elif mode == 'all':
        csvFile = open('../snap_data/confirm_all.csv', 'r')
    reader = csv.reader(csvFile)

    data = []
    for item in reader:
        if begin <= item[0] <= end:
            item[1] = float(item[1])
            if if_norm == 'train':
                item[1] = (item[1] - max_min_param['new'][0]) / (max_min_param['new'][1] - max_min_param['new'][0])
            data.append(item[1])
    csvFile.close()
    return data
    
    
def read_unemployment_pred(mode='self'):
    begin = later_n_date(alldates[0], -7)
    end = alldates[-1]
    
    if mode == 'self':
        csvFile = open('../snap_data/ump_pred.csv', 'r')
    elif mode == 'all':
        csvFile = open('../snap_data/ump_all.csv', 'r')

    reader = csv.reader(csvFile)

    data = []
    for item in reader:
        if begin <= item[0] <= end:
            item[1] = float(item[1])
            item[1] = (item[1] - max_min_param['ump'][0]) / (max_min_param['ump'][1] - max_min_param['ump'][0])
            data.append(item[1])
    csvFile.close()
    return data


def read_mobility_pred(mode='self'):
    if mode == 'self':
        csvFile = open('../snap_data/mobility_pred.csv', 'r')
    elif mode == 'all':
        csvFile = open('../snap_data/mobility_all.csv', 'r')
    reader = csv.reader(csvFile)

    begin_date = later_n_date(alldates[0], -6)
    data = []

    for item in reader:
        if reader.line_num == 1:
            continue
        if begin_date <= item[0] <= alldates[-1]:
            for i in range(1, len(item)):
                item[i] = (float(item[i]) - max_min_param['m{}'.format(i-1)][0]) / (max_min_param['m{}'.format(i-1)][1] - max_min_param['m{}'.format(i-1)][0])
            data.append(item[1:])
    
    csvFile.close()
    return np.array(data)

