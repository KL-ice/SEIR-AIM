import os
import csv
import json
import math
import random
import numpy as np
import pandas as pd
import datetime as dt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import r2_score


def fix_random_seed():
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)


def read_ump(date_train, date_test):
    csvFile = open('unemployment_web.csv', 'r')
    reader = csv.reader(csvFile)

    data = {}
    for row in reader:
        data[row[0]] = float(row[1])
    train_data, test_data = [], []
    for d in date_train:
        train_data.append(data[d])
    for d in date_test:
        test_data.append(data[d])
    
    return train_data, test_data


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


def ARIMA_Func(train_data):
    arima_model = ARIMA(train_data, order=(2,0,0))
    arima_result = arima_model.fit()
    pred = arima_result.predict(start=9, end=10)
    return list(pred)

def SVR_Func(train_data):
    train_x, train_y = [], []
    for i in range(len(train_data)):
        train_x.append(train_data[i:i+2])
        if i + 2 < len(train_data):
            train_y.append(train_data[i+2])

    svr_model = SVR()
    svr_model.fit(np.array(train_x[:len(train_y)]), np.array(train_y))
    pred1 = svr_model.predict(np.array(train_x[-2:-1]))
    pred2 = svr_model.predict(np.array([[train_data[-1], pred1[0]]]))
    pred = [pred1[0], pred2[0]]
    return pred

def RF_Func(train_data):
    train_x, train_y = [], []
    for i in range(len(train_data)):
        train_x.append(train_data[i:i+2])
        if i + 2 < len(train_data):
            train_y.append(train_data[i+2])
    
    rf_model = RandomForestRegressor()
    rf_model.fit(np.array(train_x[:len(train_y)]), np.array(train_y))
    pred1 = rf_model.predict(np.array(train_x[-2:-1]))
    pred2 = rf_model.predict(np.array([[train_data[-1], pred1[0]]]))
    pred = [pred1[0], pred2[0]]
    return pred

class ARNN_model(nn.Module):
    # 定义全连接模型
    def __init__(self):
        super(ARNN_model, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=2, out_features=64, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=64, out_features=32, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=32, out_features=1, bias=True))

    def forward(self, x):
        output = self.fc(x)
        return output

def ARNN_Func(train_data):
    train_x, train_y = [], []
    for i in range(len(train_data)):
        train_x.append(train_data[i:i+2])
        if i + 2 < len(train_data):
            train_y.append(train_data[i+2])

    arnn_model = ARNN_model().cuda()
    arnn_optim = optim.Adam([{'params':arnn_model.parameters(), 'lr': 1e-3, 'weight_decay': 5e-5}])
    x = torch.FloatTensor(np.array(train_x[:len(train_y)])).cuda()
    y = torch.FloatTensor(np.array(train_y)).cuda()

    x_train, x_val = x[:-2, :], x[-2:, :]
    y_train, y_val = y[:-2], y[-2:]

    min_loss = 1000
    for i in range(200):
        arnn_model.train()
        y_pred = arnn_model(x_train)
        arnn_optim.zero_grad()
        loss = F.mse_loss(y_pred.view(-1, 1), y_train.view(-1, 1))
        loss.backward()
        arnn_optim.step()

        arnn_model.eval()
        y_pred = arnn_model(x_val)
        loss_val = F.mse_loss(y_pred.view(-1, 1), y_val.view(-1, 1)).item()
        if loss_val < min_loss:
            min_loss = loss_val
            torch.save(arnn_model, './ump_arnn.pkl') 

    arnn_model=torch.load('./ump_arnn.pkl')
    arnn_model.eval()
    pred1 = arnn_model(torch.FloatTensor(np.array(train_x[-2:-1])).cuda()).detach().cpu().numpy()
    pred2 = arnn_model(torch.FloatTensor(np.array([[train_data[-1], pred1[0][0]]])).cuda()).detach().cpu().numpy()
    pred = [pred1[0][0], pred2[0][0]]
    return pred

class MLP_model(nn.Module):
    # 定义全连接模型
    def __init__(self):
        super(MLP_model, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=2, out_features=64, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=64, out_features=32, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=32, out_features=2, bias=True))

    def forward(self, x):
        output = self.fc(x)
        return output

def MLP_Func(train_data):
    train_x, train_y = [], []
    for i in range(len(train_data)):
        if i+2 <= len(train_data):
            train_x.append(train_data[i:i+2])
        if i + 4 <= len(train_data):
            train_y.append(train_data[i+2:i+4])

    mlp_model = MLP_model().cuda()
    mlp_optim = optim.Adam([{'params':mlp_model.parameters(), 'lr': 1e-3, 'weight_decay': 5e-5}])
    x = torch.FloatTensor(np.array(train_x[:len(train_y)])).cuda()
    y = torch.FloatTensor(np.array(train_y)).cuda()

    x_train, x_val = x[:-2, :], x[-2:, :]
    y_train, y_val = y[:-2], y[-2:]

    min_loss = 1000
    for i in range(200):
        mlp_model.train()
        y_pred = mlp_model(x_train)
        mlp_optim.zero_grad()
        loss = F.mse_loss(y_pred.view(-1, 2), y_train.view(-1, 2))
        loss.backward()
        mlp_optim.step()

        mlp_model.eval()
        y_pred = mlp_model(x_val)
        loss_val = F.mse_loss(y_pred.view(-1, 2), y_val.view(-1, 2)).item()
        if loss_val < min_loss:
            min_loss = loss_val
            torch.save(mlp_model, './ump_mlp.pkl') 

    mlp_model=torch.load('./ump_mlp.pkl')
    mlp_model.eval()
    pred = list(mlp_model(torch.FloatTensor(np.array(train_x[-1:])).cuda()).detach().cpu().numpy()[0])
    # print(pred)
    # pred2 = mlp_model(torch.FloatTensor(np.array([[train_data[-1], pred1[0][0]]])).cuda()).detach().cpu().numpy()
    # pred = [pred1[0][0], pred2[0][0]]
    return pred


def GBDT_Func(train_data):
    train_x, train_y = [], []
    for i in range(len(train_data)):
        train_x.append(train_data[i:i+2])
        if i + 2 < len(train_data):
            train_y.append(train_data[i+2])

    gbr_model = GradientBoostingRegressor(n_estimators=20)
    gbr_model.fit(np.array(train_x[:len(train_y)]), np.array(train_y))
    pred1 = gbr_model.predict(np.array(train_x[-2:-1]))
    pred2 = gbr_model.predict(np.array([[train_data[-1], pred1[0]]]))
    pred = [pred1[0], pred2[0]]
    return pred

def DES_Func(train_data):
    # print(train_data)
    a = 0.9
    S1 = [train_data[0]]
    S2 = [train_data[0]]
    for i in range(1, len(train_data)):
        S1.append(train_data[i] + (1-a)*S1[i-1])
        S2.append(a*S1[i] + (1-a)*S2[i-1])
    
    at = 2*S1[-1]-S2[-1]
    bt = (a/(1-a)) * (S1[-1] - S2[-1])

    pred1 = at + bt
    pred2 = at + 2*bt

    return [pred1, pred2]


fix_random_seed()
dataset = [str(x.date()) for x in pd.date_range(start=dt.date(2020, 4, 11), end=dt.date(2020, 11, 28), freq='W-SAT')]

all_pred, all_test = [], []
for model_weeks in range(0, 25, 2):
    date_train = dataset[model_weeks:9+model_weeks]
    date_test = dataset[9+model_weeks:11+model_weeks]
    # print(date_train, date_test)
    train_data, test_data = read_ump(date_train, date_test)
    # pred = ARIMA_Func(train_data)
    # pred = SVR_Func(train_data)
    # pred = RF_Func(train_data)
    # pred = MLP_Func(train_data)
    # pred = ARNN_Func(train_data)
    # pred = GBDT_Func(train_data)
    pred = DES_Func(train_data)

    all_pred = all_pred + pred
    all_test = all_test + test_data
    # break
cal_mape(all_pred[:-1], all_test)