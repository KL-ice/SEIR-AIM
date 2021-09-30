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

def read_7models_data(idx):
    li = [0, 4, 8, 12, 16, 20, 24]
    dataset = [str(x.date()) for x in pd.date_range(dt.date(2020, 4, 11), dt.date(2021, 4, 28))]

    pred_ump = []
    pred_mob = []

    alldates = dataset[0:57]

    ump_pred = read_pred_unemployment('../week{}/result/not_open_ave/ump.csv'.format(0), alldates)
    pred_ump = pred_ump + ump_pred

    confirm_pred = read_pred_unemployment('../week{}/result/not_open_ave/m{}.csv'.format(0, idx), alldates)
    pred_mob = pred_mob + confirm_pred

    for model_weeks in range(0, 25, 2):
        alldates = dataset[57+model_weeks*7:57+model_weeks*7+14]
        if model_weeks == 24:
            alldates = dataset[57+model_weeks*7:57+model_weeks*7+30]

        ump_pred = read_pred_unemployment('../week{}/result/not_open_ave/ump.csv'.format(model_weeks), alldates)
        pred_ump = pred_ump + ump_pred

        confirm_pred = read_pred_unemployment('../week{}/result/not_open_ave/m{}.csv'.format(model_weeks, idx), alldates)
        pred_mob = pred_mob + confirm_pred
    
    return pred_ump, pred_mob

def paint_sca(pred_ump, pred_mob, idx):
    name = ['retail_and_recreation', 'grocery_and_pharmacy', 'parks', 'transit_stations', 'workplaces', 'residential']
    
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
 
    for i in range(len(pred_ump)):
        if i == 0:
            p3 = plt.scatter(pred_mob[i], round(float(pred_ump[i]), 3), alpha=1, s=80, c=color[idx], label=name[idx])
        else:
            # break
            p3 = plt.scatter(pred_mob[i], round(float(pred_ump[i]), 3), alpha=1, s=80, c=color[idx])
    # for i in range(len(pred_ump)):
    #     p3 = plt.scatter(pred_mob[i], pred_ump[i], alpha=1, s=80)

    


if __name__ == '__main__':
    idx = 0
    plt.figure(figsize=(12, 8))
    for idx in range(6):
        pred_ump, pred_mob = read_7models_data(idx)
        paint_sca(pred_ump, pred_mob, idx)
    
    plt.xlabel('Mobility', fontsize=23)
    plt.ylabel('Unemployment Rate', fontsize=23)

    ax = plt.gca()  
    ax.grid(axis='x',which='major',color= 'gray',alpha = 0.2)
    ax.grid(axis='y',which='major',color= 'gray',alpha = 0.2)
    plt.xlim((-60, 90))

    plt.tick_params(labelsize=18)
    plt.legend(loc = 'lower right', fontsize=16, frameon=True)
    plt.title('{} and Unemployment Rate'.format('mobility'), fontsize=21)
    plt.savefig('./mob_ump/m_all.jpg', dpi=1200, pad_inches=0.0)

