import csv
import pandas as pd
import numpy as np

def read_daily(path):
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0] >= "2020-02-27"]
    day = df[0].tolist()
    df = np.array(df).tolist()
    return df, day

if __name__ == '__main__':
    daily_row, day_row = read_daily('./daily/daily_row.csv')

    # for week in [0, 4, 8, 12, 16, 20, 24]:
    for week in range(0, 25, 2):
        daily_week, day_week = read_daily('../week{}/result/not_open_ave/daily.csv'.format(week))
        idx = day_row.index(day_week[0])

        rows = daily_row[:idx] + daily_week
        csvFile = open('./daily/daily_week{}.csv'.format(week), 'w', newline='')
        writer = csv.writer(csvFile)
        writer.writerows(rows)
        csvFile.close()
    # print(daily_row, day_row)