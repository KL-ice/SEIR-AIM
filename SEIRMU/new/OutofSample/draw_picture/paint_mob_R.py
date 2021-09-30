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
import os
import csv
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_pred_confirm(path, alldates):
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0] >= alldates[0]]
    df = df[df[0] <= alldates[-1]]
    return df[1].tolist()


def read_7models_data(idx):
    li = [0, 4, 8, 12, 16, 20, 24]
    dataset = [str(x.date()) for x in pd.date_range(dt.date(2020, 4, 11), dt.date(2021, 4, 28))]

    pred_ump = []
    pred_mob = []

    alldates = dataset[0:57]

    ump_pred = read_pred_confirm('./R/daily_week{}_R0.csv'.format(0), alldates)
    pred_ump = pred_ump + ump_pred

    mob_pred = read_pred_confirm('../week{}/result/not_open_ave/m{}.csv'.format(0, idx), alldates)
    pred_mob = pred_mob + mob_pred

    for model_weeks in range(0, 25, 2):
        alldates = dataset[57+model_weeks*7:57+model_weeks*7+14]
        if model_weeks == 24:
            alldates = dataset[57+model_weeks*7:57+model_weeks*7+30]

        ump_pred = read_pred_confirm('./R/daily_week{}_R0.csv'.format(model_weeks), alldates)
        pred_ump = pred_ump + ump_pred

        mob_pred = read_pred_confirm('../week{}/result/not_open_ave/m{}.csv'.format(model_weeks, idx), alldates)
        pred_mob = pred_mob + mob_pred
    
    return pred_ump, pred_mob

def paint_sca(pred_ump, pred_mob, idx):
    name = ['retail_and_recreation', 'grocery_and_pharmacy', 'parks', 'transit_stations', 'workplaces', 'residential']
    color = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", '#FC1D1D', '#FD8D2C', '#FDED2C', '#A4F721', '#4CFC2C', '#17FE5E', '#1EF4C3', '#20C8FB', '#2F6CF5', '#340FFE', '#A61DFB', '#F412E3', '#FD1F85', '#F63636']
    color = ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462']
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
 
    for i in range(len(pred_ump)):
        if i == 0:
            p3 = plt.scatter(pred_mob[i], round(float(pred_ump[i]), 3), alpha=1, s=80, c=color[idx], label=name[idx])
        else:
            # break
            p3 = plt.scatter(pred_mob[i], round(float(pred_ump[i]), 3), alpha=1, s=80, c=color[idx])

    

if __name__ == '__main__':
    idx = 0
    plt.figure(figsize=(12, 8))
    for idx in range(6):
        pred_ump, pred_mob = read_7models_data(idx)
        paint_sca(pred_ump, pred_mob, idx)
    plt.xlabel('Mobility', fontsize=23)
    plt.ylabel('Rt', fontsize=23)

    ax = plt.gca()  
    ax.grid(axis='x',which='major',color= 'gray',alpha = 0.2)
    ax.grid(axis='y',which='major',color= 'gray',alpha = 0.2)
    plt.xlim((-90, 70))

    plt.tick_params(labelsize=18)
    plt.legend(loc = 'upper left', fontsize=16, frameon=False)
    plt.title('{} and Rt'.format('Mobility'), fontsize=21)

    plt.savefig('./mob_Rt/m_all.jpg'.format(idx), dpi=1200, pad_inches=0.0)
    


