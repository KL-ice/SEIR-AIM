import os
import csv
import math
import random
import pickle
import torch
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.font_manager import *

from config import *

def split_train_set(data, loss_weight):
    train_data, test_data = {}, {}
    for k in data:
        train_len, test_len = 0, 0
        train_data[k] = []
        test_data[k] = []
        for i in range(len(data[k])):
            if loss_weight[i] == 1:
                train_data[k].append(data[k][i])
                train_len += 1
            elif loss_weight[i] == 0:
                test_data[k].append(data[k][i])
                test_len += 1
        train_data['len'] = train_len
        test_data['len'] = test_len
    # print(train_data['len'], test_data['len'])
    return train_data, test_data


def fix_random_seed():
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)


def max_min_norm(x, param):
    _min = np.array(param[0])
    _max = np.array(param[1])
    return (x - _min) / (_max - _min)

def fintune_rate(optimizer, lr_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_rate

def fintune_rate_for_mobility(optimizer, lr_rate, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (lr_rate**epoch)

def later_n_date(datestr, n):
    datestr = dt.datetime.strptime(datestr, '%Y-%m-%d').date()
    datestr = (datestr+dt.timedelta(days=n)).strftime("%Y-%m-%d")
    return datestr

def cut_param(model, idx):
    for name,parameters in model.named_parameters():
        if name in ['fc.0.weight', 'fc.3.weight', 'policy_c', 'blm_param']:
            if name == 'policy_c':
                if idx != 5:
                    parameters.data.clamp_(-np.inf, -1e-20)
                else:
                    parameters.data.clamp_(1e-20, np.inf)
            elif name == 'blm_param':
                parameters.data.clamp_(-np.inf, -1e-20)
            else:
                parameters.data.clamp_(1e-20, np.inf)


def paint(y1, y2, title):
    fill_range = [dt.datetime.strptime(alldates[0], '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)]
    pic_x1 = fill_range[:len(y1)]
    pic_x2 = fill_range[:len(y2)]
    plt.figure(figsize=(8,6))
    plt.plot(pic_x1, y1, 'b+-')
    plt.plot(pic_x2,y2, "k*:")
    plt.grid("True")
    plt.legend(['prediction', 'truth'])
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.savefig('../figs/{}.jpg'.format(title))


# def paint_m(y1, y2, title, params):
#     alldates = [dt.datetime.strptime('2020-04-11', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)]
#     pic_x1 = alldates[:len(y1)]
#     pic_x2 = alldates[:len(y2)]
#     for i in range(len(y1)):
#         y1[i] = y1[i] * (params[1] - params[0]) + params[0]
#     for i in range(len(y2)):
#         y2[i] = y2[i] * (params[1] - params[0]) + params[0]
#     plt.figure(figsize=(8,6))
#     plt.plot(pic_x1, y1, 'b+-')
#     plt.plot(pic_x2,y2, "k*:")
#     plt.grid("True")
#     plt.legend(['prediction', 'truth'])
#     plt.title(title)
#     plt.gcf().autofmt_xdate()

def str_to_dt(datestr):
    return dt.datetime.strptime(datestr, '%Y-%m-%d').date()

def dt_to_str(date):
    return str(date)
