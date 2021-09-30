import os
import json
import numpy as np
import pandas as pd
import datetime as dt

# 8周训练，1周样本外测试，多个model

# model_weeks = int(os.path.abspath(os.path.join(os.getcwd(), "../")).split('/')[-1][4:])
model_weeks = 0

dataset = [str(x.date()) for x in pd.date_range(dt.date(2020, 4, 11), dt.date(2020, 11, 28))]
alldates = dataset[model_weeks*7:57+model_weeks*7]
print('alldates = ', len(alldates), alldates)

loss_weight = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1])

max_min_param = {}

all_max_min = {}
with open('../../row_data/max_min_num.json', 'r') as f:
    all_max_min = json.load(f)
# max_min_param = all_max_min[str(model_weeks)]
max_min_param = all_max_min[str(0)]
print(max_min_param)

# max_min_param['m0'] = [-17.142857142857142, -13.142857142857142]
# max_min_param['m1'] = [-7.428571428571429, -2.857142857142857]
# max_min_param['m2'] = [1.0, 51.857142857142854]
# max_min_param['m3'] = [-32.57142857142857, -26.857142857142858]
# max_min_param['m4'] = [-30.285714285714285, -25.142857142857142]
# max_min_param['m5'] = [6.285714285714286, 8.0]
# max_min_param['ump'] = [4.1, 8.928571428571429]
# max_min_param['inf'] = [0, 0.025891149930393653]
# max_min_param['new'] = [23545.0, 128036.0]