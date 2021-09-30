import os
import json
import numpy as np
import pandas as pd
import datetime as dt

# 8周训练，1周样本外测试，多个model

model_weeks = int(os.path.abspath(os.path.join(os.getcwd(), "../")).split('/')[-1][4:])
# model_weeks = 0

dataset = [str(x.date()) for x in pd.date_range(dt.date(2020, 4, 11), dt.date(2020, 11, 28))]
alldates = dataset[(model_weeks-2)*7:57+model_weeks*7]
print('alldates = ', len(alldates), alldates)

# loss_weight = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1])
loss_weight = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])

max_min_param = {}

# all_max_min = {}
# with open('../../row_data/max_min_num.json', 'r') as f:
#     all_max_min = json.load(f)
# max_min_param = all_max_min[str(model_weeks)]
max_min_param = {'m0': [-30.142857142857142, -11.428571428571429], 'm1': [-6.428571428571429, 2.9285714285714284], 'm2': [6.0, 62.714285714285715], 'm3': [-41.42857142857143, -26.0], 'm4': [-39.57142857142857, -28.714285714285715], 'm5': [8.285714285714286, 14.571428571428571], 'inf': [0, 0.03399206488656227], 'ump': [11.2, 15.6], 'new': [17190.0, 76588.0]}
