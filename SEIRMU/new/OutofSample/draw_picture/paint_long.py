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

from read_data import *
from config import *
import matplotlib.font_manager

def paint_ump(pred_data, truth_data, bggin_date, idx):

    lenth_pred = len(pred_data)
    lenth_truth = len(truth_data)

    x_pred = [dt.datetime.strptime(bggin_date, '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(0, 1000, 7)][:lenth_pred]
    y_pred = pred_data
    x_truth = [dt.datetime.strptime(bggin_date, '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(0, 1000, 7)][:lenth_truth]
    y_truth = truth_data

    #调整图片大小
    plt.figure(figsize=(8,6))


    # 设定坐标轴刻度朝内
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    tick_params(which='both',top='on',bottom='on',left='on',right='on')

    #设定字体
    plt.rc('font',family='Times New Roman')
    #设定标题字体
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    ax = plt.gca()  
    # ax.set_ylim(6000000,130000000)  
    # ax.set_ylim(6000000,60000000)  
    # 设置网格线
    ax.grid(axis='x',which='major',color= 'gray',alpha = 0.4)
    ax.grid(axis='y',which='major',color= 'gray',alpha = 0.4)

    # 设置坐标轴边框宽度
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    
    plt.plot(x_pred,y_pred,label="Prediction Unemployment Rate",color="#D95319",linewidth=3)
    plt.plot(x_truth,y_truth,label="Unemployment Rate by Database",color="#0072BD",linewidth=3)

    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlabel("Date",fontsize=18)
    plt.ylabel("Unemployment Rate",fontsize=18)

    plt.title("Unemployment Rate",fontsize=20)
    # plt.gcf().autofmt_xdate()
    # ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.legend(loc='center left', bbox_to_anchor=(0.23, 0.78), ncol=1,borderaxespad=0.3, edgecolor='gray',labelspacing=0.1,handletextpad=0.7,columnspacing=0.3,handlelength=1.7,borderpad=0.2,
                fancybox=True,frameon=True,fontsize=16,shadow=False, framealpha=0.5)
    # plt.show()
    plt.savefig('./result_long/ump_{}.svg'.format(idx), dpi=1200, pad_inches=0.0)

def read_pred_unemployment(path, alldates):
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

def read_7models_data():
    li = [0, 4, 8, 12, 16, 20]
    # li = [12]
    dataset = [str(x.date()) for x in pd.date_range(dt.date(2020, 4, 11), dt.date(2021, 4, 28))]

    pred_ump, truth_ump = [], []
    pred_con, truth_con = [], []

    # alldates = dataset[0:57]

    # ump_pred = read_pred_unemployment('../week{}/result/not_open_ave/ump.csv'.format(0), ump_date, alldates)

    for model_weeks in li:
        alldates = dataset[model_weeks*7:57+model_weeks*7+90]
        if model_weeks == 24:
            alldates = dataset[model_weeks*7:57+model_weeks*7+90]

        
        ump_pred = read_pred_unemployment('../week{}/result/not_open_ave/ump.csv'.format(model_weeks), alldates)
        ump_truth, ump_date = read_truth_unemployment('../row_data/unemployment_web.csv', alldates[:57])
        # pred_ump = pred_ump + ump_pred
        paint_ump(ump_pred, ump_truth, alldates[0], model_weeks)

read_7models_data()