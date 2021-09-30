import os
import csv
import math
import json
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import r2_score

# 0 8 20 

# li = [0, 4, 8, 12, 16, 20, 24]
# li = []
# model_weeks = li[1]

dataset = [str(x.date()) for x in pd.date_range(dt.date(2020, 4, 11), dt.date(2020, 11, 28))]
# for i in li:
    # model_weeks = i
# print(alldates)
# print(dataset[model_weeks*7:57+model_weeks*7])

def mape(truth, pred):
    # print(truth, pred)
    err = 0
    # score = r2_score(truth, pred, multioutput='raw_values')
    # print(score)
    # print(truth, pred)
    for i in range(0, len(truth)):
        err += abs(truth[i]-pred[i])# / truth[i]
        # err += abs(truth[i]-pred[i])
        # err += 
    return err / (len(truth))

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

def read_pred_unemployment(path, dates):
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0].isin(dates)]
    return df[1].tolist()

def read_truth_unemployment(path):
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0] >= alldates[0]]
    df = df[df[0] <= alldates[-1]]
    return df[1].tolist(), df[0].tolist()


def read_pred_confirm(path):
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0] >= alldates[0]]
    df = df[df[0] <= alldates[-1]]
    return df[1].tolist()

def read_truth_confirm(path):
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0] >= alldates[0]]
    df = df[df[0] <= alldates[-1]]
    return df[1].tolist()

if __name__ == '__main__':

    ump_mape, com_mape = [], []

    ump_pred_all, ump_truth_all = [], []
    com_pred_all, com_truth_all = [], []
    
    for model_weeks in range(0, 25, 2):
        alldates = dataset[57+model_weeks*7:57+model_weeks*7+14]
        # print(dataset[model_weeks*7:57+model_weeks*7])


        ump_truth, ump_date = read_truth_unemployment('./row_data/unemployment_web.csv')
        ump_pred = read_pred_unemployment('./week{}/result/not_open_ave/ump.csv'.format(model_weeks), ump_date)

        confirm_truth = read_truth_confirm('./row_data/confirm.csv')
        confirm_pred = read_pred_confirm('./week{}/result/not_open_ave/acc confirm.csv'.format(model_weeks))


        # print(ump_pred)
        ump_mape.append(mape(ump_truth, ump_pred))
        ump_truth_all = ump_truth_all + ump_truth#[:1]
        ump_pred_all = ump_pred_all + ump_pred#[:1]

        print(len(confirm_truth))
        com_mape.append(mape(confirm_truth, confirm_pred))
        com_truth_all = com_truth_all + confirm_truth#[:7]
        com_pred_all = com_pred_all + confirm_pred#[:7]

    # print(np.mean(ump_mape))
    print(ump_mape)
    print(np.mean(ump_mape), np.mean(com_mape))
    # print(len(ump_pred_all), len(ump_truth_all))
    cal_mape(ump_pred_all, ump_truth_all)
    # print(len(com_pred_all), len(com_truth_all))
    cal_mape(com_pred_all, com_truth_all)
