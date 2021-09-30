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


def paint_confirm(pred_data, truth_data):

    lenth_pred = len(pred_data)
    lenth_truth = len(truth_data)

    x_pred = [dt.datetime.strptime('2020-09-26', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_pred]
    y_pred = pred_data

    x_truth = [dt.datetime.strptime('2020-09-26', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_truth]
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
    ax.set_ylim(6000000,60000000)  
    # 设置网格线
    ax.grid(axis='x',which='major',color= 'gray',alpha = 0.4)
    ax.grid(axis='y',which='major',color= 'gray',alpha = 0.4)

    # 设置坐标轴边框宽度
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    
    plt.plot(x_pred,y_pred,label="Prediction Confirmed Cases",color="#D95319",linewidth=3)
    plt.plot(x_truth,y_truth,label="Confirmed Cases by Database",color="#0072BD",linewidth=3)

    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlabel("Date",fontsize=18)
    plt.ylabel("Confirmed Cases",fontsize=18)

    # plt.title("Confirmed Cases",fontsize=20)
    # plt.gcf().autofmt_xdate()
    # ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.legend(loc='center left', bbox_to_anchor=(0.03, 0.78), ncol=1,borderaxespad=0.3, edgecolor='gray',labelspacing=0.1,handletextpad=0.7,columnspacing=0.3,handlelength=1.7,borderpad=0.2,
                fancybox=True,frameon=True,fontsize=20,shadow=False, framealpha=0.5)
    # plt.show()
    plt.savefig('./result/confirm.jpg', dpi=1200, pad_inches=0.0)


def paint_unemployment(pred_data, truth_data):

    lenth_pred = len(pred_data)
    lenth_truth = len(truth_data)

    x_pred = [dt.datetime.strptime('2020-09-26', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_pred]
    y_pred = pred_data

    x_truth = [dt.datetime.strptime('2020-09-26', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_truth]
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
    # ax.set_ylim(0,13000000)  
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
    plt.gcf().autofmt_xdate()
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.legend(loc='center right',bbox_to_anchor=(1, 0.7), ncol=1,borderaxespad=0.3, edgecolor='gray',labelspacing=0.1,handletextpad=0.7,columnspacing=0.3,handlelength=1.7,borderpad=0.1,
                fancybox=True,frameon=True,fontsize=20,shadow=False, framealpha=0.5)
    # plt.show()
    plt.savefig('./result/unemployment.svg', dpi=1200, pad_inches=0.0)


def paint_inf(pred_data, truth_data):

    lenth_pred = len(pred_data)
    lenth_truth = len(truth_data)

    x_pred = [dt.datetime.strptime('2020-09-26', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_pred]
    y_pred = pred_data

    x_truth = [dt.datetime.strptime('2020-09-26', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_truth]
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
    # ax.set_ylim(0,13000000)  
    # 设置网格线
    ax.grid(axis='x',which='major',color= 'gray',alpha = 0.4)
    ax.grid(axis='y',which='major',color= 'gray',alpha = 0.4)

    # 设置坐标轴边框宽度
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    
    plt.plot(x_pred,y_pred,label="Prediction Infection Rate",color="#D95319",linewidth=3)
    plt.plot(x_truth,y_truth,label="Infection Rate by Database",color="#0072BD",linewidth=3)

    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlabel("Date",fontsize=18)
    plt.ylabel("Infection Rate",fontsize=18)

    plt.title("Infection Rate",fontsize=20)
    plt.gcf().autofmt_xdate()
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.legend(loc='center right',bbox_to_anchor=(1, 0.7), ncol=1,borderaxespad=0.3, edgecolor='gray',labelspacing=0.1,handletextpad=0.7,columnspacing=0.3,handlelength=1.7,borderpad=0.1,
                fancybox=True,frameon=True,fontsize=20,shadow=False, framealpha=0.5)
    # plt.show()
    
    plt.subplots_adjust(left=0.15, top=None, wspace=None, hspace=None)
    plt.savefig('./result/infection_rate.svg', dpi=1200, pad_inches=0.0)


def paint_Rt(pred_data, truth_data):

    lenth_pred = len(pred_data)
    lenth_truth = len(truth_data)

    x_pred = [dt.datetime.strptime('2020-03-20', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_pred]
    y_pred = pred_data

    x_truth = [dt.datetime.strptime('2020-03-20', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_truth]
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
    # ax.set_ylim(0,13000000)  
    # 设置网格线
    ax.grid(axis='x',which='major',color= 'gray',alpha = 0.4)
    ax.grid(axis='y',which='major',color= 'gray',alpha = 0.4)

    # 设置坐标轴边框宽度
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    
    plt.plot(x_pred,y_pred,label="Prediction Rt",color="#D95319",linewidth=3)
    plt.plot(x_truth,y_truth,label="Rt by Database",color="#0072BD",linewidth=3)

    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlabel("Date",fontsize=18)
    plt.ylabel("Rt",fontsize=18)

    plt.title("Reproduction Number", fontsize=20)
    plt.gcf().autofmt_xdate()
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.legend(loc='center right',bbox_to_anchor=(0.98, 0.73),ncol=1,borderaxespad=0.3, edgecolor='gray',labelspacing=0.1,handletextpad=0.7,columnspacing=0.3,handlelength=1.7,borderpad=0.1,
                fancybox=True,frameon=True,fontsize=20,shadow=False, framealpha=0.5)
    # plt.show()


def paint_blm_confirm(pred_data):

    lenth_pred = len(pred_data)

    x_pred = [dt.datetime.strptime('2020-06-27', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_pred]
    y_pred = pred_data

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
    
    plt.plot(x_pred,y_pred,label="prediction Rt",color="#D95319",linewidth=3)

    
    plt.tick_params(labelsize=18)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlabel("Date",fontsize=23, fontweight='bold')
    plt.ylabel("Increased confirmed \ncases caused by BLM",fontsize=23, fontweight='bold')

    # plt.title("Increased Cumulative Confirmed Cases Caused by BLM", fontsize=20)
    plt.gcf().autofmt_xdate()
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    # plt.legend(loc='upper right',ncol=1,borderaxespad=0.3, edgecolor='gray',labelspacing=0.1,handletextpad=0.7,columnspacing=0.3,handlelength=1.7,borderpad=0.1,
                # fancybox=True,frameon=True,fontsize=20,shadow=False, framealpha=0.5)
    # plt.show()
    plt.savefig('./result/blm confirm.svg', dpi=1200, pad_inches=0.05, bbox_inches = 'tight')


def paint_blm_unemployment(pred_data):

    lenth_pred = len(pred_data)

    x_pred = [dt.datetime.strptime('2020-06-27', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_pred]
    y_pred = pred_data

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
    
    plt.plot(x_pred,y_pred,label="prediction Rt",color="#D95319",linewidth=3)

    
    plt.tick_params(labelsize=18)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlabel("Date",fontsize=23, fontweight='bold')
    plt.ylabel("Increased unemployment \nrate caused by BLM",fontsize=23, fontweight='bold')

    # plt.title("Increased Unemployment Rate Caused by BLM", fontsize=20)
    plt.gcf().autofmt_xdate()
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    # plt.legend(loc='upper right',ncol=1,borderaxespad=0.3, edgecolor='gray',labelspacing=0.1,handletextpad=0.7,columnspacing=0.3,handlelength=1.7,borderpad=0.1,
                # fancybox=True,frameon=True,fontsize=20,shadow=False, framealpha=0.5)
    # plt.show()
    plt.savefig('./result/blm unemployment.svg', dpi=1200, pad_inches=0.05, bbox_inches = 'tight')


def paint_blm_index(pred_data):

    lenth_pred = len(pred_data)

    x_pred = [dt.datetime.strptime('2020-05-20', '%Y-%m-%d').date() + dt.timedelta(days = x) for x in range(1000)][:lenth_pred]
    y_pred = pred_data

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
    ax.set_ylim(0,1)  
    # 设置网格线
    # ax.grid(axis='x',which='major',color= 'gray',alpha = 0.4)
    # ax.grid(axis='y',which='major',color= 'gray',alpha = 0.4)

    # 设置坐标轴边框宽度
    # ax.spines['bottom'].set_color('#0072BD') 
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(1)
    # ax.spines['right'].set_linewidth(1)
    ax.spines['right'].set_color('none')
    
    plt.plot(x_pred,y_pred,label="prediction Rt",color="#0072BD",linewidth=3, clip_on=False, zorder=3)

    
    plt.tick_params(labelsize=18)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlabel("Date",fontsize=23, fontweight='bold')
    plt.ylabel("\nBLM Index",fontsize=23, fontweight='bold')

    # plt.title("BLM Index", fontsize=20)
    plt.gcf().autofmt_xdate()
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    # plt.legend(loc='upper right',ncol=1,borderaxespad=0.3, edgecolor='gray',labelspacing=0.1,handletextpad=0.7,columnspacing=0.3,handlelength=1.7,borderpad=0.1,
                # fancybox=True,frameon=True,fontsize=20,shadow=False, framealpha=0.5)
    # plt.show()
    plt.subplots_adjust(left=0.1, bottom=0.25, right=None, top=None)
    plt.savefig('./result/blm index.svg', dpi=1200, pad_inches=0.05, bbox_inches = 'tight')


if __name__ == "__main__":

    # paint confirm cases
    # pred_confirm = read_pred_confirm(pred_confirm_path)
    # truth_confirm = read_truth_confirm(truth_confirm_path)
    # paint_confirm(pred_confirm, truth_confirm)

    # paint unemployment rate
    # pred_unemployment = read_pred_unemployment(pred_unemployment_path)
    # truth_unemployment = read_truth_unemployment(truth_unemployment_path)
    # paint_unemployment(pred_unemployment, truth_unemployment)

    # paint unemployment rate
    # pred_Rt = read_pred_Rt(pred_Rt_path)
    # truth_Rt = read_truth_Rt(truth_Rt_path)
    # paint_Rt(pred_Rt, truth_Rt)

    # paint blm different confirm
    pred_blm_confirm = read_blm_confirm(blm_confirm_path, noblm_pred_confirm_path)
    print(pred_blm_confirm)
    paint_blm_confirm(pred_blm_confirm)

    # paint blm different unemployment
    pred_blm_unemployment = read_blm_unemployment(blm_unemployment_path, noblm_pred_unemployment_path)
    print(pred_blm_unemployment)
    paint_blm_unemployment(pred_blm_unemployment)

    blm_index = read_blm_inedx(blm_index_path)
    paint_blm_index(blm_index)

    # pred_inf = [0.00954041164368391, 0.010497291572391987, 0.011120161972939968, 0.011480739340186119, 0.011714714579284191, 0.011915579438209534, 0.01210733037441969, 0.012298648245632648, 0.012515188194811344, 0.012772245332598686, 0.013041410595178604, 0.013296606950461864, 0.013511604629456997, 0.013682755641639233, 0.013837314210832119, 0.013980587013065815, 0.01415819302201271, 0.014362465590238571, 0.014596445485949516, 0.014872822910547256, 0.015195727348327637, 0.015555099584162235, 0.015946270897984505, 0.016329947859048843, 0.016726627945899963, 0.017150476574897766, 0.017611948773264885, 0.01811842992901802, 0.0186744574457407, 0.019233373925089836, 0.019821060821413994, 0.02044534869492054, 0.021105162799358368, 0.0217897966504097, 0.022479863837361336, 0.023150412365794182, 0.02383909747004509, 0.024510089308023453, 0.025131208822131157, 0.025675132870674133, 0.026120556518435478, 0.026453059166669846, 0.026665957644581795, 0.026752926409244537, 0.026725631207227707, 0.026600774377584457, 0.02639753557741642, 0.02613612823188305, 0.025836963206529617, 0.02552018314599991, 0.025211604312062263, 0.024926593527197838, 0.0246753953397274, 0.024463104084134102, 0.02429005317389965, 0.02415289357304573, 0.024046216160058975, 0.02396765537559986, 0.023909378796815872, 0.02386578358709812, 0.023832842707633972, 0.02380770817399025, 0.023788345977663994, 0.023773277178406715, 0.023760177195072174, 0.02374868467450142, 0.023738589137792587, 0.023729639127850533, 0.02372164838016033, 0.023714477196335793, 0.023708004504442215, 0.023702135309576988, 0.023696795105934143, 0.023691847920417786, 0.023687273263931274, 0.023683026432991028, 0.023679085075855255, 0.02367541566491127, 0.023671990260481834, 0.023668792098760605, 0.023665815591812134, 0.02366304025053978, 0.0236604493111372, 0.02365802973508835, 0.023655762895941734, 0.023653637617826462, 0.023651644587516785, 0.023649776354432106, 0.02364801988005638, 0.023646365851163864, 0.02364480309188366, 0.023643335327506065, 0.02364194579422474, 0.02364063635468483, 0.02363939955830574, 0.023638226091861725, 0.023637115955352783, 0.023636065423488617, 0.023635070770978928, 0.02363412454724312, 0.02363322302699089, 0.023632371798157692, 0.023631559684872627, 0.023630788549780846, 0.02363005466759205, 0.023629354313015938, 0.023628689348697662, 0.023628052324056625, 0.023627446964383125, 0.023626862093806267, 0.023626312613487244, 0.02362578734755516, 0.023625284433364868, 0.02362479828298092, 0.02362433634698391, 0.02362389862537384, 0.023623475804924965, 0.023623069748282433, 0.023622682318091393, 0.023622315376996994, 0.023621955886483192, 0.023621613159775734, 0.023621289059519768, 0.023620974272489548, 0.023620672523975372, 0.02362038940191269, 0.023620113730430603, 0.023619843646883965, 0.02361958660185337, 0.023619340732693672, 0.02361910417675972, 0.023618875071406364, 0.023618659004569054, 0.02361844852566719, 0.023618245497345924, 0.023618051782250404, 0.02361786924302578, 0.023617688566446304, 0.023617513477802277, 0.023617349565029144, 0.02361719124019146, 0.023617036640644073, 0.023616887629032135, 0.023616742342710495, 0.0236166063696146, 0.023616474121809006]
    # truth_inf = [0.010669372654577562, 0.010743240777903455, 0.011088392290809719, 0.011265271501965275, 0.011462000790084845, 0.011562583441632683, 0.01156611761364354, 0.011653536320369123, 0.012039351765481387, 0.012346229607414416, 0.012462875007794968, 0.012730140084263297, 0.012777295531378526, 0.012847473469267073, 0.013122772616106913, 0.01342940903331596, 0.013825499055621235, 0.014240617405671027, 0.014371340671943785, 0.014751034383418807, 0.014813940578132718, 0.01513148986078562, 0.01564035684584562, 0.015997450537550747, 0.016503159356639333, 0.016873487000925773, 0.01725682576936934, 0.017514821092231986, 0.01788690190946341, 0.01866813324533322, 0.019538106392953718, 0.020196194105407496, 0.02105420048583124, 0.02160278622977868, 0.022103806515040448, 0.02257332422559926, 0.02320140776025004, 0.02374017975137745, 0.024515457371323938, 0.0247350483605454, 0.025252417470816486, 0.02557000287228027, 0.025888138607007626, 0.026453451187721327, 0.027004958855033203, 0.027219782122427445, 0.027054776460013908, 0.026819453020947334, 0.026500476140760496, 0.026123419498499372, 0.025989503849476524, 0.025796745229456727, 0.02558611066731834, 0.025133127836314745, 0.024320086338151945, 0.023748778904985637, 0.0231924095516537]
    # paint_inf(pred_inf, truth_inf)
    
