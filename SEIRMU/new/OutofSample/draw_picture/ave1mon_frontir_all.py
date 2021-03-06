import os
import csv
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

date_str = '2020-06-30'
alldates = [str(x.date()) for x in pd.date_range(dt.date(2020, 6, 1), dt.date(2020, 11, 1))]


NNN = 1

plt.rc('font', family = 'Times New Roman', size=12)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def see_oneday(begin_d, date_str, num):
    not_open = []

    csvFile = open('../week4/result/not_open_ave/acc confirm.csv', 'r')
    reader = csv.reader(csvFile)
    for item in reader:
        if item[0] == date_str:
            not_open.append(float(item[1]))
    csvFile.close()

    csvFile = open('../week4/result/not_open_ave/ump.csv', 'r')
    reader = csv.reader(csvFile)
    for item in reader:
        if item[0] == date_str:
            not_open.append(float(item[1]))
    csvFile.close()

    # print(not_open)
    confirm = []
    ump = []
    names = []
    for i in range(1, 256):
        # print('{:08b}'.format(i) )
        # if not (NNN <= bin(i).count('1') <= NNN):
        #     continue
        # if '{:08b}'.format(i) == '00000000':
        #     continue
        # print('./{}/{:08b}/acc confirm.csv'.format(begin_d, i))
        csvFile = open('../week4/result/csvs_diff_begin_frontier_ave/{}/{:08b}/acc confirm.csv'.format(begin_d, i), 'r')
        reader = csv.reader(csvFile)

        for item in reader:
            if item[0] == date_str:
                confirm.append((float(item[1]) - not_open[0]) / 1)
                break

        csvFile.close()

        csvFile = open('../week4/result/csvs_diff_begin_frontier_ave/{}/{:08b}/ump.csv'.format(begin_d, i), 'r')
        reader = csv.reader(csvFile)

        for item in reader:
            if item[0] == date_str:
                # print(i, item[1], not_open[1])
                ump.append((-float(item[1]) + not_open[1]) / 1)
                break

        csvFile.close()

        names.append('{:08b}'.format(i))


    return confirm, ump, names


def get_allday(begin_d):
    alldates = [str(x.date()) for x in pd.date_range(dt.date(2020, 6, 1), dt.date(2020, 11, 1))]

    date = alldates[alldates.index(begin_d)+14]

    confirm, ump, names = see_oneday(begin_d, date, 1)

    return confirm, ump, names

def deep(dd):

    r = int(dd[1:3],16) + 80
    g = int(dd[3:5],16) + 80
    b = int(dd[5:7], 16) + 80
    rgb = str(r)+','+str(g)+','+str(b)

    RGB = rgb.split(',')            # ???RGB??????????????????
    cc = '#'
    for i in RGB:
        num = int(i)
        # ???R???G???B???????????????16???????????????????????????  hex() ???????????????10?????????????????????16?????????????????????????????????
        cc += str(hex(num))[-2:].replace('x', '0').upper()
    return cc


def write_csv(acc_confirm, acc_ump, acc_names):
    rows = []
    # print('xxxx', acc_ump, acc_confirm)
    for i in range(len(acc_names)):
        row = [acc_names[i], acc_ump[i], acc_confirm[i]]
        rows.append(row)
    csvFile = open('./policy.csv', 'w', newline='')
    writer = csv.writer(csvFile)
    writer.writerows(rows)
    csvFile.close()


def main():
    dir_date = [str(x.date()) for x in pd.date_range(dt.date(2020, 6, 1), dt.date(2020, 6, 15))]

    acc_confirm, acc_ump, acc_names = [], [], []


    for d in dir_date:
        confirm, ump, names = get_allday(d)
        acc_confirm.append(confirm)
        acc_ump.append(ump)
        acc_names = names

    acc_confirm = np.array(acc_confirm)
    acc_ump = np.array(acc_ump)

    print(acc_confirm.shape)
    acc_confirm = np.average(acc_confirm, axis=0)
    acc_ump = np.average(acc_ump, axis=0)
    print(acc_confirm.shape)
    print(acc_confirm, acc_ump)

    write_csv(acc_confirm, acc_ump, acc_names)



    names = ['C1: School closing','C2: Workplace closing','C3: Cancel public events','C4: Restrictions on gatherings','C5: Close public transport','C6: Stay at home requirements','C7: Restrictions on internal movement','C8: International travel controls']
    color = ['#FC1D1D', '#FD8D2C', '#FDED2C', '#A4F721', '#4CFC2C', '#17FE5E', '#1EF4C3', '#20C8FB', '#2F6CF5', '#340FFE', '#A61DFB', '#F412E3', '#FD1F85', '#F63636']
    for i in range(13):
        color[i] = deep(color[i])

    # con_max, con_min = np.max(acc_confirm), np.min(acc_confirm)
    # ump_max, ump_min = np.max(acc_ump), np.min(acc_ump)

    txt = ['C1', 'C2' ,'C3' ,'C4','C5','C6', 'C7', 'C8']


    confirm = acc_confirm  
    ump = acc_ump

    disp = np.ones_like(confirm)
    for i in range(confirm.shape[0]):
        for j in range(confirm.shape[0]):
            if confirm[i] > confirm[j] and ump[i] < ump[j]:
                disp[i] = 0


    plt.figure(figsize=(12, 8))

    for i in range(confirm.shape[0]):
        # if disp[i] != 1:
        #     continue

        # disp_name = ''
        # for j in range(8):
        #     if acc_names[i][j] == '1':
        #         if disp_name == '':
        #             disp_name += txt[j]
        #         else:
        #             disp_name = disp_name + '+' + txt[j]

        # print(acc_names[i])
        p3 = plt.scatter(confirm[i], ump[i], alpha=1, s=80)
        # print(disp_name)

        

    # for i in range(8):
        # p3 = plt.scatter(0, 0, alpha=0, label=names[i])
        # plt.text(0, 0.67-i*0.05, names[i], fontsize=18)

    plt.xlabel('Increments in cumulative confirmed cases', fontsize=23, fontweight='bold')
    plt.ylabel('Increments in employment rate', fontsize=23, fontweight='bold')
    # plt.xlim((-130000, 130000))
    # plt.ylim((-0.45, 0.45))
    # plt.xlim((-150, 3500))
    # plt.ylim((-0.03, 0.83))
    ax = plt.gca()  
    ax.grid(axis='x',which='major',color= 'gray',alpha = 0.2)
    ax.grid(axis='y',which='major',color= 'gray',alpha = 0.2)

    plt.tick_params(labelsize=18)
    plt.legend(loc = 'upper left', fontsize=16, frameon=False)
    # plt.title('Open policy from 0601 to 0615, \nobserve the changes after 14 days(all policy combinations)', fontsize=21)

    # plt.subplots_adjust(left=0.01, bottom=0.1, right=1, top=None, wspace=None, hspace=None)
    plt.savefig('./result/all policy.svg', dpi=1200,  pad_inches=0.05, bbox_inches = 'tight')
    # plt.close('all')


if __name__ == "__main__":

    # for i in range(8):
    #     NNN = i
    #     main()
    main()
    