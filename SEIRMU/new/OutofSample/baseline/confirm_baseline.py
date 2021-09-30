import os
import csv
import copy
import json
import random
import pickle
import subprocess
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.font_manager import *
import scipy.integrate as spi
from scipy.optimize import minimize
from scipy.signal import savgol_filter

import argparse
import glob
import random
import sys
import time
import torch
import math

from sklearn.metrics import r2_score

def fix_random_seed():
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)


def read_data(file_name):
    csvFile = open(file_name, 'r')
    reader = csv.reader(csvFile)

    data = {}
    data_titles = []
    for item in reader:
        if reader.line_num == 1:
            data_titles = item
            continue
        for i in range(1, len(item)):
            data[data_titles[i]] = int(item[i])
    csvFile.close()
    return data


def get_state_data(fill_range):
    confirm = read_data('./US_confirm.csv')
    cured = read_data('./US_cured.csv')
    death = read_data('./US_death.csv')

    data_confirm, data_remove = [], []

    for d in fill_range:
        data_confirm.append(confirm[d])
        data_remove.append(cured[d] + death[d])
    
    return np.array(data_confirm), np.array(data_remove)

def get_E0(begin_date):
    csvFile = open('./E.csv', 'r')
    reader = csv.reader(csvFile)
    _E0 = 0
    for item in reader:
        if item[0] <= begin_date:
            _E0 += float(item[1])
    csvFile.close()

    return _E0



class SeirTraining():
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight
        self.loss_weight[self.loss_weight==2] = 0
        return 

    def SEIR(self, INPUT, t, alpha, beta, gamma):
        # S E I R
        Y = np.zeros((4))

        alpha = max(alpha, 0)
        beta = max(beta, 0)
        gamma = max(gamma, 0)

        Y[0] = - beta * INPUT[0] * INPUT[2]
        Y[1] = beta * INPUT[0] * INPUT[2] - alpha * INPUT[1]
        Y[2] = alpha * INPUT[1] - gamma * INPUT[2]
        Y[3] = gamma * INPUT[2]
        return Y

    def SEIISR_loss(self, TRUE, PRED):
        return np.sum(np.square(((TRUE[:, 0] - TRUE[:, 1]) - (PRED [:, 2])) * self.loss_weight)) + np.sum(np.square((TRUE[:, 1] - PRED[:, 3]) * self.loss_weight))

    def optim_fun(self, args):
        INPUT, t_range, TRUE = args
        v = lambda x: self.SEIISR_loss(TRUE, spi.odeint(self.SEIR, (INPUT[0], INPUT[1], INPUT[2], INPUT[3]), t_range, args = (x[0], x[1], x[2])))
        return v

    def predict_Ndays(self, init_S, infect_data, remove_data, N, E0=None, x0_init = None):
        S0 = init_S
        I0 = infect_data[0] - remove_data[0]
        R0 = remove_data[0]
        S0 = S0 - I0 - R0 - E0
        INPUT = [S0, I0 * 2. if E0 is None else E0, I0, R0]
        TRUE = np.array([infect_data,
                        remove_data])
        TRUE = TRUE.T

        S0 = 1. if S0 == 0 else S0

        INPUT = [S0, INPUT[1], I0, R0]
        # print('E0 = {}'.format(INPUT[1]))
        
        t_range = np.arange(0, len(TRUE), 1)
        
        x0 = x0_init
        # print('x0', x0, INPUT)
        RES = minimize(self.optim_fun((INPUT, t_range, TRUE)), x0, method = 'Nelder-Mead')
        # print('res.x', RES.x)

        # 拟合现在
        x = RES.x
        # x = x0
        t_range = np.arange(0.0, N, 1)

        RES0 = spi.odeint(self.SEIR, (INPUT[0], INPUT[1], INPUT[2], INPUT[3]), t_range[:len(infect_data)], args = (x[0], x[1], x[2]))
        
        # 预测第一周
        Input = RES0[-1].copy()
        Input[3] = remove_data[-1]
        Input[2] = infect_data[-1] - remove_data[-1]
        RES1 = spi.odeint(self.SEIR, Input, t_range[len(infect_data)-1:], args = (x[0], x[1], x[2]))
        
        # 合并
        RES = np.concatenate((RES0, RES1[1:, ]))
        PRED = RES[:, 2] + RES[:, 3] # I + R
        return PRED, {'S': RES[:, 0], 'E': RES[:, 1], 'I': RES[:, 2], 'R': RES[:, 3], 'I+R': PRED, 'ACTUAL_ALL': PRED+RES[:, 1]}, x


class SirTraining():
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight
        self.loss_weight[self.loss_weight==2] = 0
        return 

    def SIR(self, INPUT, t, alpha, beta):
        # S E I R
        Y = np.zeros((3))

        alpha = max(alpha, 0)
        beta = max(beta, 0)

        Y[0] = - alpha * INPUT[0] * INPUT[1]
        Y[1] = alpha * INPUT[0] * INPUT[1] - beta * INPUT[1]
        Y[2] = beta * INPUT[1]
        return Y

    def SEIISR_loss(self, TRUE, PRED):
        return np.sum(np.square(((TRUE[:, 0] - TRUE[:, 1]) - (PRED [:, 1])) * self.loss_weight)) + np.sum(np.square((TRUE[:, 1] - PRED[:, 2]) * self.loss_weight))

    def optim_fun(self, args):
        INPUT, t_range, TRUE = args
        v = lambda x: self.SEIISR_loss(TRUE, spi.odeint(self.SIR, (INPUT[0], INPUT[1], INPUT[2]), t_range, args = (x[0], x[1])))
        return v

    def predict_Ndays(self, init_S, infect_data, remove_data, N, x0_init = None):
        S0 = init_S
        I0 = infect_data[0] - remove_data[0]
        R0 = remove_data[0]
        S0 = S0 - I0 - R0
        INPUT = [S0, I0, R0]
        TRUE = np.array([infect_data,
                        remove_data])
        TRUE = TRUE.T

        S0 = 1. if S0 == 0 else S0

        INPUT = [S0, I0, R0]
        # print('E0 = {}'.format(INPUT[1]))
        
        t_range = np.arange(0, len(TRUE), 1)
        
        x0 = x0_init
        # print('x0', x0, INPUT)
        RES = minimize(self.optim_fun((INPUT, t_range, TRUE)), x0, method = 'Nelder-Mead', options = {'disp': True})
        # print('res.x', RES.x)

        # 拟合现在
        x = RES.x
        # x = x0
        t_range = np.arange(0.0, N, 1)

        RES0 = spi.odeint(self.SIR, (INPUT[0], INPUT[1], INPUT[2]), t_range[:len(infect_data)], args = (x[0], x[1]))
        
        # 预测第一周
        Input = RES0[-1].copy()
        Input[2] = remove_data[-1]
        Input[1] = infect_data[-1] - remove_data[-1]
        RES1 = spi.odeint(self.SIR, Input, t_range[len(infect_data)-1:], args = (x[0], x[1]))
        
        # 合并
        RES = np.concatenate((RES0, RES1[1:, ]))
        PRED = RES[:, 1] + RES[:, 2] 
        return PRED


def train_seir(alldates):
    loss_weight = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1])


    pop_scale = {}
    pop_scale['United States'] = [328239523, 30000, 100, 1.3, 1]
    params = {}

    # 发病率,传染率,移除率
    params['United States'] = np.array([3.37885689e-02, 3.41702513e-09, 7.71553139e-03])
    
    seir_training = SeirTraining(copy.deepcopy(loss_weight))
    init_S = pop_scale['United States'][0] / pop_scale['United States'][4]

    infect_data, remove_data = get_state_data(alldates)
    # print(infect_data.shape)


    x0 = params['United States']
        

    E0 = get_E0(alldates[0])
    NN = 57+14
    PRED_ALL, data_all, x0 = seir_training.predict_Ndays(init_S, infect_data, remove_data, NN, x0_init=x0, E0=E0)
    
    # write_pred_result(PRED_ALL)

    # print(list(PRED_ALL[-14:]))
    t_range_subdt = [dt.datetime.strptime(alldates[0], '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(NN)]
    disp_days = len(infect_data)
    plt.figure(figsize=(8,6))
    plt.plot(t_range_subdt[:disp_days], PRED_ALL[:disp_days], 'b+-')
    plt.plot(t_range_subdt[:len(infect_data)], infect_data, "k*:")
    plt.grid("True")
    plt.legend(["pred", "GT"])
    plt.title(u'{} confirm (data {})'.format('US', alldates[-1]))
    plt.xlabel('Date')
    plt.ylabel('Case')
    plt.gcf().autofmt_xdate()
    plt.savefig('./{}1_accum_confirmed_day{}.jpg'.format('US', alldates[-1]), dpi=200)
    return list(PRED_ALL[-14:])


def train_sir(alldates):
    loss_weight = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1])


    fix_random_seed()
    pop_scale = {}
    pop_scale['United States'] = [328239523, 30000, 100, 1.3, 1]
    params = {}

    # 发病率,传染率,移除率
    params['United States'] = np.array([3.41702513e-09, 7.71553139e-03])
    
    sir_training = SirTraining(copy.deepcopy(loss_weight))
    init_S = pop_scale['United States'][0] / pop_scale['United States'][4]

    infect_data, remove_data = get_state_data(alldates)

    x0 = params['United States']
        
    NN = 57+14
    PRED_ALL = sir_training.predict_Ndays(init_S, infect_data, remove_data, NN, x0_init=x0)
    
    t_range_subdt = [dt.datetime.strptime(alldates[0], '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(NN)]
    disp_days = len(infect_data)
    plt.figure(figsize=(8,6))
    plt.plot(t_range_subdt[:disp_days], PRED_ALL[:disp_days], 'b+-')
    plt.plot(t_range_subdt[:len(infect_data)], infect_data, "k*:")
    plt.grid("True")
    plt.legend(["pred", "GT"])
    plt.title(u'{} confirm (data {})'.format('US', alldates[-1]))
    plt.xlabel('Date')
    plt.ylabel('Case')
    plt.gcf().autofmt_xdate()
    plt.savefig('./{}1_accum_confirmed_day{}.jpg'.format('US', alldates[-1]), dpi=200)

    # print(list(PRED_ALL[-14:]))
    return list(PRED_ALL[-14:])


def cal_mape(pred, truth):
    err = 0
    mae, mape, rmse, r2 = 0, 0, 0, 0
    for i in range(0, len(pred)):
        mae += abs(pred[i] - truth[i])  # mae
        mape += abs(pred[i] - truth[i]) / truth[i]  #mape
        rmse += math.pow(pred[i] - truth[i], 2)

    r2 = r2_score(truth, pred)
    # print(err / (len(pred)))
    mae = mae / len(pred)
    mape = mape / len(pred)
    rmse = math.pow(rmse / len(pred), 0.5)
    print("mae: {}, mape: {}, rmse: {}, r2: {}".format(mae, mape, rmse, r2))


if __name__ == '__main__':
    fix_random_seed()

    dataset = [str(x.date()) for x in pd.date_range(dt.date(2020, 4, 11), dt.date(2020, 11, 28))]

    all_pred, all_truth = [], []
    for model_weeks in range(0, 25, 2):
        alldates = dataset[model_weeks*7:57+model_weeks*7]
        pred = train_seir(alldates)
        # pred = train_sir(alldates)
        alldates = dataset[57+model_weeks*7:57+model_weeks*7+14]
        confirm_truth, _ = get_state_data(alldates)
        
        all_pred = all_pred + pred[:7]
        all_truth = all_truth + list(confirm_truth)[:7]
        # break
    # print(len(all_truth), len(all_pred[:-7]))
    # cal_mape(all_pred[:-7], all_truth)
    print(len(all_pred), len(all_truth))
    cal_mape(all_pred, all_truth)