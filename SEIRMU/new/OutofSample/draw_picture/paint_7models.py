import os
import csv
import math
import json
import numpy as np
import pandas as pd
import datetime as dt
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.font_manager import *
from pylab import *
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import AutoMinorLocator

def read_pred_unemployment(path, dates, alldates):
    sat = [str(x.date()) for x in pd.date_range(dt.date(2020, 4, 11), dt.date(2021, 4, 28), freq='W-SAT')]
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0].isin(alldates)]
    df = df[df[0].isin(sat)]
    return df[1].tolist()

def read_truth_unemployment(path, alldates):
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0] >= alldates[0]]
    df = df[df[0] <= alldates[-1]]
    return df[1].tolist(), df[0].tolist()


def read_pred_confirm(path, alldates):
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0] >= alldates[0]]
    df = df[df[0] <= alldates[-1]]
    return df[1].tolist()

def read_truth_confirm(path, alldates):
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0] >= alldates[0]]
    df = df[df[0] <= alldates[-1]]
    return df[1].tolist()

def read_7models_data():
    li = [0, 4, 8, 12, 16, 20, 24]
    dataset = [str(x.date()) for x in pd.date_range(dt.date(2020, 4, 11), dt.date(2021, 4, 28))]

    pred_ump, truth_ump = [], []
    pred_con, truth_con = [], []

    alldates = dataset[0:57]

    ump_truth, ump_date = read_truth_unemployment('../row_data/unemployment_web.csv', alldates)
    ump_pred = read_pred_unemployment('../week{}/result/not_open_ave/ump.csv'.format(0), ump_date, alldates)
    pred_ump = pred_ump + ump_pred
    truth_ump = truth_ump + ump_truth

    confirm_truth = read_truth_confirm('../row_data/confirm.csv', alldates)
    confirm_pred = read_pred_confirm('../week{}/result/not_open_ave/acc confirm.csv'.format(0), alldates)
    pred_con = pred_con + confirm_pred
    truth_con = truth_con + confirm_truth

    for model_weeks in range(0, 25, 2):
        alldates = dataset[57+model_weeks*7:57+model_weeks*7+14]
        if model_weeks == 24:
            alldates = dataset[57+model_weeks*7:57+model_weeks*7+60]

        ump_truth, ump_date = read_truth_unemployment('../row_data/unemployment_web.csv', alldates)
        ump_pred = read_pred_unemployment('../week{}/result/not_open_ave/ump.csv'.format(model_weeks), ump_date, alldates)
        pred_ump = pred_ump + ump_pred
        truth_ump = truth_ump + ump_truth

        if model_weeks == 24:
            print(len(ump_pred))

        confirm_truth = read_truth_confirm('../row_data/confirm.csv', alldates)
        confirm_pred = read_pred_confirm('../week{}/result/not_open_ave/acc confirm.csv'.format(model_weeks), alldates)
        pred_con = pred_con + confirm_pred
        truth_con = truth_con + confirm_truth
    
    return pred_ump, truth_ump, pred_con, truth_con

def paint_confirm(pred_data, truth_data):

    lenth_pred = len(pred_data)
    lenth_truth = len(truth_data)
    print(lenth_pred, lenth_truth)

    x_pred = [dt.datetime.strptime('2020-04-11', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_pred]
    y_pred = pred_data

    x_truth = [dt.datetime.strptime('2020-04-11', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_truth]
    y_truth = truth_data

    #调整图片大小
    plt.figure(figsize=(12,5.5))


    # 设定坐标轴刻度朝内
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    tick_params(which='both',bottom='on',left='on')

    #设定字体
    plt.rc('font',family='Times New Roman')
    #设定标题字体
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    ax = plt.gca()  
    ax.set_ylim(0,35000000)  
    # xlim = [dt.datetime.strptime('2020-04-11', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_pred+10]
    # ax.set_xlim(x_truth[0],xlim[-1])
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # 设置网格线
    # ax.grid(axis='x',which='major',color= 'gray',alpha = 0.4)
    # ax.grid(axis='y',which='major',color= 'gray',alpha = 0.4)

    # 设置坐标轴边框宽度
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(1)
    # ax.spines['right'].set_linewidth(1)
    ax.spines['right'].set_color('none')
    
    plt.plot(x_pred,y_pred,label="Prediction Confirmed Cases",color="#D95319",linewidth=3)
    plt.plot(x_truth,y_truth,label="Confirmed Cases by Database",color="#0072BD",linewidth=3)

    
    plt.tick_params(labelsize=18)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlabel("Date",fontsize=23, fontweight='bold')
    plt.ylabel("The number of \nconfirmed cases in U.S.",fontsize=23, fontweight='bold')

    # plt.title("Confirmed Cases",fontsize=20)
    plt.gcf().autofmt_xdate()
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.subplots_adjust(left=0.15, wspace=None, hspace=None)

    plt.legend(loc='center left', bbox_to_anchor=(0.03, 0.78), ncol=1,borderaxespad=0.3, labelspacing=0.1,handletextpad=0.7,columnspacing=0.3,handlelength=1.7,borderpad=0.2,
                fancybox=True,frameon=False,fontsize=18,shadow=False, framealpha=0.5)
    # plt.show()

    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    plt.subplots_adjust(left=0.1, right=0.95)
    plt.savefig('./result/confirm_7models.svg', dpi=1200,  pad_inches=0.05, bbox_inches = 'tight')     


def paint_unemployment(pred_data, truth_data):

    lenth_pred = len(pred_data)
    lenth_truth = len(truth_data)
    print(lenth_pred, lenth_truth)

    x_pred = [dt.datetime.strptime('2020-04-11', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(0, 1000, 7)][:lenth_pred]
    y_pred = pred_data

    x_truth = [dt.datetime.strptime('2020-04-11', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(0, 1000, 7)][:lenth_truth]
    # print(x_truth)
    y_truth = truth_data

    #调整图片大小
    plt.figure(figsize=(12,5.5))

    # 设定坐标轴刻度朝内
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    tick_params(which='both',bottom='on',left='on')

    #设定字体
    plt.rc('font',family='Times New Roman')
    #设定标题字体
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    ax = plt.gca()  
    # ax.set_ylim(0,13000000)  
    # 设置网格线
    # ax.grid(axis='x',which='major',color= 'gray',alpha = 0.4)
    # ax.grid(axis='y',which='major',color= 'gray',alpha = 0.4)

    # 设置坐标轴边框宽度
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(1)
    # ax.spines['right'].set_linewidth(1)
    ax.spines['right'].set_color('none')
    
    plt.plot(x_pred,y_pred,label="Prediction Unemployment Rate",color="#D95319",linewidth=3)
    plt.plot(x_truth,y_truth,label="Unemployment Rate by Database",color="#0072BD",linewidth=3)

    
    plt.tick_params(labelsize=18)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlabel("Date",fontsize=23, fontweight='bold')
    plt.ylabel("\nUnemployment Rate",fontsize=23, fontweight='bold')

    plt.gcf().autofmt_xdate()
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.legend(loc='center right',bbox_to_anchor=(1, 0.85), ncol=1,borderaxespad=0.3, edgecolor='gray',labelspacing=0.1,handletextpad=0.7,columnspacing=0.3,handlelength=1.7,borderpad=0.1,
                fancybox=True,frameon=False,fontsize=18,shadow=False, framealpha=0.5)
    # plt.show()
    plt.savefig('./result/ump_7models.svg', dpi=1200, pad_inches=0.05, bbox_inches = 'tight')




if __name__ == '__main__':
    pred_ump, truth_ump, pred_con, truth_con = read_7models_data()
    x_pred = [dt.datetime.strptime('2020-04-11', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)]
    # for i in range(1, len(pred_con)):
    #     print(x_pred[i], pred_con[i]-pred_con[i-1])
    paint_confirm(pred_con, truth_con)

    paint_unemployment(pred_ump, truth_ump)
