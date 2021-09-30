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

from config import *
from utils import *
from data_reader import *

myfont = FontProperties(fname='../../../objs/simhei.ttf', size=14)


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
    confirm = read_data('../../row_data/US_confirm.csv')
    cured = read_data('../../row_data/US_cured.csv')
    death = read_data('../../row_data/US_death.csv')

    data_confirm, data_remove = [], []

    for d in fill_range:
        data_confirm.append(confirm[d])
        data_remove.append(cured[d] + death[d])
    
    return np.array(data_confirm), np.array(data_remove)


def get_all_rate(fill_range, train_iter):
    if train_iter <= 50:
        csvFile = open('../../row_data/E.csv', 'r')
    else:
        csvFile = open('../snap_data/inf_all.csv', 'r')
    reader = csv.reader(csvFile)
    
    data = []
    for item in reader:
        if item[0] in fill_range[1:]:
            rate = float(item[2])
            data.append(rate)
    csvFile.close()
    return data


class SeirTraining():
    def __init__(self, inf_rate, loss_weight):
        self.inf_rate = inf_rate
        self.loss_weight = loss_weight
        self.loss_weight[self.loss_weight==2] = 0
        return 

    def get_inf_rate(self, t):
        t = int(t)
        if t < len(self.inf_rate):
            return self.inf_rate[t]
        else:
            return self.inf_rate[-1]

    def SEIR(self, INPUT, t, alpha, beta, gamma):
        # S E I R
        Y = np.zeros((4))

        alpha = max(alpha, 0)
        beta = max(beta, 0)
        gamma = max(gamma, 0)
        it = self.get_inf_rate(t)

        Y[0] = - it*beta * INPUT[0] * INPUT[2]
        Y[1] = it*beta * INPUT[0] * INPUT[2] - alpha * INPUT[1]
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
        RES = minimize(self.optim_fun((INPUT, t_range, TRUE)), x0, method = 'Nelder-Mead', options = {'disp': True})
        # print('res.x', RES.x)

        # 拟合现在
        x = RES.x
        # x = x0
        t_range = np.arange(0.0, N, 1)

        RES0 = spi.odeint(self.SEIR, (INPUT[0], INPUT[1], INPUT[2], INPUT[3]), t_range[:len(infect_data)], args = (x[0], x[1], x[2]))

        # print('res0.shape, infect_data.shape: ', RES0.shape, len(infect_data))
        # print('res0[-1:]: ', RES0[-1, :])
        # print(np.sum(np.square((TRUE[:, 0] - TRUE[:, 1]) - (RES0 [:, 2]))) + np.sum(np.square(TRUE[:, 1] - RES0[:, 3])))
        
        # 预测第一周
        Input = RES0[-1].copy()
        Input[3] = remove_data[-1]
        Input[2] = infect_data[-1] - remove_data[-1]
        RES1 = spi.odeint(self.SEIR, Input, t_range[len(infect_data)-1:], args = (x[0], x[1], x[2]))
        
        # 合并
        RES = np.concatenate((RES0, RES1[1:, ]))
        PRED = RES[:, 2] + RES[:, 3] # I + R
        return PRED, {'S': RES[:, 0], 'E': RES[:, 1], 'I': RES[:, 2], 'R': RES[:, 3], 'I+R': PRED, 'ACTUAL_ALL': PRED+RES[:, 1]}, x


def write_pred_result(PRED_ALL):
    rows = []
    
    write_begin = later_n_date(alldates[0], -10)
    before_date = [str(x.date()) for x in pd.date_range(str_to_dt(write_begin), str_to_dt(alldates[0]))]

    infect_data, remove_data = get_state_data(before_date)
    for i in range(1, len(infect_data)):
        rows.append([before_date[i], infect_data[i] - infect_data[i-1]])
    
    # rows = []
    DPC = [PRED_ALL[i+1] - PRED_ALL[i] for i in range(len(PRED_ALL)-1)]
    for i in range(len(alldates)-1):
        row = [alldates[i+1], DPC[i]]
        rows.append(row)
    
    csvFile = open('../snap_data/confirm_pred.csv', 'w', newline='')
    writer = csv.writer(csvFile)
    writer.writerows(rows)
    csvFile.close()


def save_param(param, train_iter):
    if_all = ''
    if train_iter > 50:
        if_all = '_all'
    with open('../models/seir/seir{}_iter{}.pkl'.format(if_all, train_iter), 'wb') as f:
        pickle.dump(param, f)


def train_seir(train_iter):
    fix_random_seed()
    pop_scale = {}
    pop_scale['United States'] = [328239523, 30000, 100, 1.3, 1]
    params = {}

    # 发病率,传染率,移除率
    if train_iter == 0:
        params['United States'] = np.array([3.37885689e-02, 3.41702513e-09, 7.71553139e-03])
    else:
        if_all = ''
        if train_iter-1 > 50:
            if_all = '_all'
        with open('../models/seir/seir{}_iter{}.pkl'.format(if_all,train_iter-1), 'rb') as f:
            params['United States'] = pickle.load(f)
        # with open('../models/seir/seir_iter{}.pkl'.format(train_iter-1), 'rb') as f:
        #     params['United States'] = pickle.load(f)

    infect_data, remove_data = get_state_data(alldates)
    inf_rate = get_all_rate(alldates, train_iter) 

    seir_training = SeirTraining(inf_rate, copy.deepcopy(loss_weight))
    init_S = pop_scale['United States'][0] / pop_scale['United States'][4]

    while(True):
        try:
            x0 = params['United States']
        except:
            x0 = None

        # E0 = 659479
        E0 = get_E0()
        NN = 240
        PRED_ALL, data_all, x0 = seir_training.predict_Ndays(init_S, infect_data, remove_data, NN, x0_init=x0, E0=E0)
        print('x0', x0)
        if (x0 == params['United States']).all():
            break
        else:
            params['United States'] = x0
    
    save_param(params['United States'], train_iter)
    write_pred_result(PRED_ALL)

    ACTUAL_ALL = data_all['ACTUAL_ALL']
    
    startdate = alldates[0]
    based_time = str_to_dt(startdate)
    t_range_subdt = [based_time + dt.timedelta(days = x) for x in range(NN)]

    cityname = 'US'
    
    path = '../figs/seir/'
    if not os.path.exists(path):
        os.makedirs(path)


    disp_days = len(infect_data)
    plt.figure(figsize=(8,6))
    plt.plot(t_range_subdt[:disp_days], PRED_ALL[:disp_days], 'b+-')
    plt.plot(t_range_subdt[:len(infect_data)], infect_data, "k*:")
    plt.grid("True")
    plt.legend(["模型预测值", "官方公布值"], prop=myfont)
    plt.title(u'{} 累计确诊数预测结果 (数据截止 {})'.format(cityname, alldates[-1]), FontProperties=myfont)
    plt.xlabel('Date')
    plt.ylabel('Case')
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'{}1_accum_confirmed_day{}.jpg'.format(cityname, disp_days), dpi=200)
    
    
    disp_days = len(infect_data)
    REMOVE_ALL = data_all['R']
    print(REMOVE_ALL[0], remove_data[0])
    plt.figure(figsize=(8,6))
    plt.plot(t_range_subdt[:disp_days], PRED_ALL[:disp_days] - REMOVE_ALL[:disp_days], 'b-')
    plt.plot(t_range_subdt[:len(infect_data)], infect_data - remove_data, "k*:")
    plt.grid("True")
    plt.legend(["模型预测值", "官方公布值"], prop=myfont)
    plt.title(u'{} 现存确诊数预测结果 (数据截止 {})'.format(cityname, alldates[-1]), FontProperties=myfont)
    plt.xlabel('Date')
    plt.ylabel('Case')
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'{}2_remained_confirmed_day{}.jpg'.format(cityname, disp_days), dpi=200)
    
    disp_days = len(infect_data)
    
    plt.figure(figsize=(8,6))
    plt.plot(t_range_subdt[:disp_days], PRED_ALL[:disp_days], 'r-')
    plt.plot(t_range_subdt[:disp_days], ACTUAL_ALL[:disp_days], 'b.-')
    plt.plot(t_range_subdt[:len(infect_data)], infect_data, "k*:")
    plt.grid("True")
    plt.legend(["累计确诊数预测", '累计感染数预测', '官方公布确诊数'], prop=myfont)
    plt.title(u'{} 累计确诊/感染数预测结果 (数据截止 {})'.format(cityname, alldates[-1]), FontProperties=myfont)
    plt.xlabel('Date')
    plt.ylabel('Case')
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'{}3_累计确诊数预测_day{}.jpg'.format(cityname, disp_days), dpi=200)
    
    disp_days = len(infect_data)-1
    plt.figure(figsize=(8,6))
    plt.plot(t_range_subdt[1:][:disp_days], [PRED_ALL[i+1] - PRED_ALL[i] for i in range(len(PRED_ALL)-1)][:disp_days], 'r+-')
    plt.plot(t_range_subdt[1:][:disp_days], [ACTUAL_ALL[i+1] - ACTUAL_ALL[i] for i in range(len(ACTUAL_ALL)-1)][:disp_days], 'b.-')
    plt.plot(t_range_subdt[1:][:len(infect_data)-1], [infect_data[i+1] - infect_data[i] for i in range(len(infect_data)-1)], 'k.-')
    plt.grid("True")
    plt.legend(["日新增确诊数预测", "日新增感染数预测", '官方日新增确诊数'], prop=myfont)
    plt.title(u'{} 日新增确诊/感染数预测结果 (数据截止 {})'.format(cityname, alldates[-1]), FontProperties=myfont)
    plt.xlabel('Date')
    plt.ylabel('Case')
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'{}4_日新增确诊_day{}.jpg'.format(cityname, disp_days), dpi=200)
    
    disp_days = len(infect_data)
    plt.figure(figsize=(8,6))
    plt.plot(t_range_subdt[1:][:disp_days], [PRED_ALL[i+1] - PRED_ALL[i] for i in range(len(PRED_ALL)-1)][:disp_days], 'b+-')
    plt.plot(t_range_subdt[:len(infect_data)][1:], [infect_data[i+1] - infect_data[i] for i in range(len(infect_data)-1)], 'k.-')
    plt.grid("True")
    plt.legend(["日新增确诊数预测", '官方日新增确诊数'], prop=myfont)
    plt.title(u'{} 日新增确诊数预测结果 (数据截止 {})'.format(cityname, alldates[-1]), FontProperties=myfont)
    plt.xlabel('Date')
    plt.ylabel('Case')
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'{}5_dailynew_confirmed_day{}.jpg'.format(cityname, disp_days), dpi=200)
    
    disp_days = len(infect_data)
    REMOVE_ALL = data_all['R']
    fig3 = plt.figure(figsize=(8,6))
    ax3 = fig3.add_subplot(1,1,1)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.plot(t_range_subdt[:disp_days], REMOVE_ALL[:disp_days], 'r+-')
    plt.plot(t_range_subdt[:len(infect_data)], remove_data, "k*:")
    plt.grid("True")
    plt.legend(["累计移除数预测", '官方公布治愈+死亡数'], prop=myfont)
    plt.title(u'{} 累计移除数预测结果 (数据截止 {})'.format(cityname, alldates[-1]), FontProperties=myfont)
    plt.xlabel('Date')
    plt.ylabel('Case')
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'{}6_累计移除数预测结果_{}days.jpg'.format(cityname, disp_days))
    
    disp_days = 240
    plt.figure(figsize = (8, 6))
    plt.plot(t_range_subdt[:disp_days], PRED_ALL[:disp_days], 'r', label = '累计确诊数')
    plt.plot(t_range_subdt[:disp_days], PRED_ALL[:disp_days] - REMOVE_ALL[:disp_days], 'g', label = '当前在治患者数')
    plt.plot(t_range_subdt[:disp_days], REMOVE_ALL[:disp_days], 'tab:orange', label = '治愈或死亡人数')
    plt.plot(t_range_subdt[:len(infect_data)], infect_data, 'b.-', label = '官方公布确诊数')
    plt.legend(prop=myfont)
    plt.grid(True)
    plt.title(u'{} 预测结果 (数据截止 {})'.format(cityname, alldates[-1]), FontProperties=myfont)
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'{}7_document_{}days.jpg'.format(cityname, disp_days))


if __name__ == '__main__':
    train_seir()