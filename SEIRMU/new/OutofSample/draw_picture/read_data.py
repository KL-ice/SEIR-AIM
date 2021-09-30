import csv
import pandas as pd
import numpy as np


def read_pred_confirm(path):
    df = pd.read_csv(path, header=None, sep=',')
    return df[1].tolist()

def read_truth_confirm(path):
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0] >= "2020-09-26"]
    return df[1].tolist()


def read_pred_unemployment(path):
    df = pd.read_csv(path, header=None, sep=',')
    return df[1].tolist()

def read_truth_unemployment(path):
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0] >= "2020-09-26"]
    return df[1].tolist()


def read_pred_Rt(path):
    df = pd.read_csv(path, header=0, sep=',')[:-1]
    df['x'] = pd.to_numeric(df['x'])
    df = df[df['Unnamed: 0'] >= "2020-03-20"]
    return df['x'].tolist()

def read_truth_Rt(path):
    df = pd.read_csv(path, header=0, sep=',')[:-1]
    df['x'] = pd.to_numeric(df['x'])
    df = df[df['Unnamed: 0'] >= "2020-03-20"]
    return df['x'].tolist()


def read_blm_confirm(blm_path, pred_path):
    df_blm = pd.read_csv(blm_path, header=None, sep=',')
    df_blm = df_blm[df_blm[0] >= "2020-06-27"]
    df_blm = df_blm[df_blm[0] <= "2020-07-27"]

    df_pred = pd.read_csv(pred_path, header=None, sep=',')
    df_pred = df_pred[df_pred[0] >= "2020-06-27"]
    df_pred = df_pred[df_pred[0] <= "2020-07-27"]

    df_blm[1] = df_pred[1] - df_blm[1]

    return df_blm[1].tolist()

def read_blm_unemployment(blm_path, pred_path):
    df_blm = pd.read_csv(blm_path, header=None, sep=',')
    df_blm = df_blm[df_blm[0] >= "2020-06-27"]
    df_blm = df_blm[df_blm[0] <= "2020-07-27"]

    df_pred = pd.read_csv(pred_path, header=None, sep=',')
    df_pred = df_pred[df_pred[0] >= "2020-06-27"]
    df_pred = df_pred[df_pred[0] <= "2020-07-27"]

    df_blm[1] = df_blm[1] - df_pred[1]

    return df_blm[1].tolist()

def read_blm_inedx(path):
    df = pd.read_csv(path, header=None, sep=',')
    df = df[df[0] >= "2020-05-20"]
    df = df[df[0] <= "2020-07-05"]
    return df[1].tolist()