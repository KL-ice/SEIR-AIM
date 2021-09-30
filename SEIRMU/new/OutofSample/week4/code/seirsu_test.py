from sklearn.metrics import r2_score

from models import *
from config import *
from utils import *
from data_reader import *


fix_random_seed()

parser = argparse.ArgumentParser()
parser.add_argument('--iternum', nargs='+', type=int, default=[50, 50, 50, 50], help='ump, seir, mob, inf')
parser.add_argument('--idx', type=int, default=0, help='idx')
parser.add_argument('--comb', type=str, default='00000000', help='comb')
parser.add_argument('--date', type=str, default='2020-06-01', help='idx')
args = parser.parse_args()
idx = args.idx
begin_change_date = args.date
comb = '0' + args.comb + '000000'
comb_idx = [int(i) for i in comb]
comb_idx = torch.FloatTensor(comb_idx).cuda().view(1,-1)

print(args.iternum)

times = 0
value = 0

if_all = '_all' if args.iternum[0] >=51 else ''
ump_model = torch.load('../models/unemployment/' + 'ump_model{}_iter{}.pkl'.format(if_all, args.iternum[0])).eval()

if_all = '_all' if args.iternum[1] >=51 else ''
with open('../models/seir/' + 'seir{}_iter{}.pkl'.format(if_all, args.iternum[1]), 'rb') as f:
    seir_param = list(pickle.load(f))
seir_mod = SEIR(seir_param[0], seir_param[1], seir_param[2]).cuda()

if_all = '_all' if args.iternum[2] >=51 else ''
m0_mod = torch.load('../models/mobility/' + 'm0{}_iter{}.pkl'.format(if_all, args.iternum[2])).eval()
m1_mod = torch.load('../models/mobility/' + 'm1{}_iter{}.pkl'.format(if_all, args.iternum[2])).eval()
m2_mod = torch.load('../models/mobility/' + 'm2{}_iter{}.pkl'.format(if_all, args.iternum[2])).eval()
m3_mod = torch.load('../models/mobility/' + 'm3{}_iter{}.pkl'.format(if_all, args.iternum[2])).eval()
m4_mod = torch.load('../models/mobility/' + 'm4{}_iter{}.pkl'.format(if_all, args.iternum[2])).eval()
m5_mod = torch.load('../models/mobility/' + 'm5{}_iter{}.pkl'.format(if_all, args.iternum[2])).eval()

if_all = '_all' if args.iternum[3] >=50 else ''
infect_net = torch.load('../models/infection/' + 'infect{}_iter{}.pkl'.format(if_all, args.iternum[3])).eval()

def get_new_confirm7(new, before7, mu, sigma):
    daily = torch.stack(new, 0)
    daily = daily[:, 2] + daily[:, 3]
    out = []

    l = min(len(daily)-1, 7)  
    for i in range(7-l):
        out.append(before7[l+i])

    for i in range(l):
        idx = len(daily) - l + i
        out.append(daily[idx] - daily[idx-1])
    #print(out)
    for i in range(len(out)):
        out[i] = (out[i] - mu) / (sigma - mu)
    
    out = torch.FloatTensor(out).cuda().view(1, -1)
    #print(out)

    return out

def get_ump7(daily, before7):
    out = []
    l = min(len(daily), 7)  
    for i in range(7-l):
        out.append(before7[l+i])

    for i in range(l):
        idx = len(daily) - l + i
        out.append(daily[idx])
    
    # print('out:', out)
    # print('before:', before7)
    out = torch.FloatTensor(out).cuda().view(1, -1)
    return out

def get_mob7(daily, before7):
    out = []
    l = min(len(daily), 7)  
    for i in range(7-l):
        out.append(before7[l+i])

    for i in range(l):
        idx = len(daily) - l + i
        out.append(daily[idx])
    
    out = torch.FloatTensor(out).cuda().view(1, 7, -1)
    return out
   

def test_lstm(N, x, m0, u0, i0, control, new7, ump7, mob7, last7, new_mu, new_sigma, inf_mu, inf_sigma, ump_data, mobility, oxc_data, infect_rate):

    global times, value

    out = [x]
    mob_out = []
    ump_out = []
    inf_out = []
    true_inf = []
    fill_range = [str(x.date()) for x in pd.date_range(str_to_dt(alldates[0]), dt.date(2021, 7, 10))]

    for i in range(1, N):

        # predict today's mobility ------------------------------------------------------------------------------

        _input = get_new_confirm7(out, new7, new_mu, new_sigma)
        _ump = get_ump7(ump_out, ump7)
        
        _oxc = torch.FloatTensor(oxc_data[i]).cuda().view(1, -1)

        if i < len(alldates) and alldates[i] == begin_change_date:
            times += 1
            #value = _oxc[:, idx] / 7
            # value = torch.FloatTensor([0.1]).view(1 , 1).cuda() / 7
            value = torch.where(comb_idx == 1, _oxc, torch.zeros_like(_oxc)) / 7
            # print(value)


        # if i >= len(alldates) or alldates[i] >= begin_change_date:
            # _oxc = torch.where((_oxc - times * value) < 0, torch.zeros_like(_oxc), (_oxc - times * value))
            # _oxc[:, 0] = 0 # for blm
            # times += 1
            # print(_oxc)
        
        # print(_oxc[:, :])
        m0 = m0_mod(_input, control, _ump, _oxc[:, 1:]).item()
        m1 = m1_mod(_input, control, _ump, _oxc[:, 1:]).item()
        m2 = m2_mod(_input, control, _ump, _oxc[:, 1:]).item()
        m3 = m3_mod(_input, control, _ump, _oxc[:, 1:]).item()
        m4 = m4_mod(_input, control, _ump, _oxc[:, 1:]).item()
        m5 = m5_mod(_input, control, _ump, _oxc).item()
        m_today = [m0, m1, m2, m3, m4, m5]
        
        # m_today[0] = mobility[i][0]
        # m_today[1] = mobility[i][1]
        # m_today[2] = mobility[i][2]
        # m_today[3] = mobility[i][3]
        # m_today[4] = mobility[i][4]
        # m_today[5] = mobility[i][5]

        mob_out.append(m_today)


        # -------------------------------------------------------------------------------------------------------
        # predict today's unemployment rate ---------------------------------------------------------------------

        _mob = get_mob7(mob_out[1:], mob7)
        if i <= 7:
            ump_day = ump_model(_mob, control, torch.FloatTensor([last7[i-1]]).cuda().view(1, 1)).item()
        else:
            ump_day = ump_model(_mob, control, torch.FloatTensor([ump_out[i-8]]).cuda().view(1, 1)).item()
        # if i >= len(alldates) or alldates[i] >= begin_change_date:
        #     ump_day-=0.000035
        ump_out.append(ump_day)

        # -------------------------------------------------------------------------------------------------------
        # predict today's infection rate ------------------------------------------------------------------------

        inf = infect_net(_mob, control)
        inf_out.append(inf.item())
        # inf = torch.FloatTensor([infect_rate[i]]).cuda()

        



        # if i < len(alldates) and  alldates[i] == '2020-06-09':
        #     print(_input, _oxc, _ump, control)
        #     print(_mob, ump_day, inf.item())
            
        # -------------------------------------------------------------------------------------------------------
        # predict today's confirm number ------------------------------------------------------------------------

        
        inf = inf * (inf_sigma - inf_mu) + inf_mu 
        if inf < 0:
            inf = torch.FloatTensor([0]).cuda()
        true_inf.append(inf.item())

        y = seir_mod(out[-1], inf)
        out.append(y)
        if fill_range[i] == '2020-07-25':
            # print(y[2]+y[3]-3969668.0)
            print(3969973.5 - y[2]-y[3])
            print(3969668.0 - y[2]-y[3])

        # -------------------------------------------------------------------------------------------------------
    out = torch.stack(out, 0)
    mob_out = np.stack(mob_out)
    # print(out)
    # print(true_inf)
    return out, mob_out, inf_out, true_inf, ump_out

def write_csv(data, name):
    fill_range = [str(x.date()) for x in pd.date_range(str_to_dt(alldates[0]), dt.date(2021, 7, 10))]
    csvFile = open(name, 'w', newline='')
    writer = csv.writer(csvFile)
    for i in range(len(data)):
        row = [fill_range[i], data[i]]
        writer.writerow(row)
    csvFile.close()

def write_mobility(out):
    rows = []

    csvFile = open('../../row_data/mobility_truth.csv', 'r')
    reader = csv.reader(csvFile)
    for item in reader:
        if reader.line_num == 1:
            continue
        if item[4] <= alldates[0]:
            rows.append(item[4:])
    csvFile.close()

    csvFile = open('../snap_data/mobility_all.csv', 'w', newline='')
    writer = csv.writer(csvFile)
    # print(len(alldates), out.shape)

    for i in range(1, len(alldates)):
        for idx in range(6):
            out[i-1][idx] = out[i-1][idx] * ((max_min_param['m{}'.format(idx)][1] - max_min_param['m{}'.format(idx)][0])) + max_min_param['m{}'.format(idx)][0]
        row = [alldates[i]] + list(out[i-1])
        rows.append(row)
    writer.writerows(rows)    
    csvFile.close()

def write_inf(out):
    rows = []

    csvFile = open('../../row_data/E.csv', 'r')
    reader = csv.reader(csvFile)
    for item in reader:
        if item[0] <= alldates[0]:
            rows.append(item)
    csvFile.close()

    csvFile = open('../snap_data/inf_all.csv', 'w', newline='')
    writer = csv.writer(csvFile)

    for i in range(1, len(alldates)):
        out[i-1] = out[i-1] * ((max_min_param['inf'][1] - max_min_param['inf'][0])) + max_min_param['inf'][0]
        row = [alldates[i], out[i-1], out[i-1]] 
        rows.append(row)
    writer.writerows(rows)    
    csvFile.close()

def write_confirm(out):
    rows = []

    csvFile = open('../snap_data/confirm_pred.csv', 'r')
    reader = csv.reader(csvFile)
    for item in reader:
        if item[0] <= alldates[0]:
            rows.append(item)
    csvFile.close()

    csvFile = open('../snap_data/confirm_all.csv', 'w', newline='')
    writer = csv.writer(csvFile)

    for i in range(1, len(alldates)):
        row = [alldates[i], out[i-1]] 
        rows.append(row)
    writer.writerows(rows)    
    csvFile.close()

def write_unemployment(out):
    rows = []

    csvFile = open('../../row_data/unemploy.csv', 'r')
    reader = csv.reader(csvFile)
    for item in reader:
        if item[0] <= alldates[0]:
            rows.append(item)
    csvFile.close()

    csvFile = open('../snap_data/ump_all.csv', 'w', newline='')
    writer = csv.writer(csvFile)

    for i in range(1, len(alldates)):
        out[i-1] = out[i-1] * ((max_min_param['ump'][1] - max_min_param['ump'][0])) + max_min_param['ump'][0]
        row = [alldates[i], out[i-1]] 
        rows.append(row)
    writer.writerows(rows)    
    csvFile.close()


def main():
    new_mu, new_sigma = max_min_param['new'][0], max_min_param['new'][1]
    inf_mu, inf_sigma = max_min_param['inf'][0], max_min_param['inf'][1]
    
    control = read_control()
    oxc_data = read_oxc_onehot(end_date='2021-02-20')
    # print(len(oxc_data))


    infect_rate = read_infect_rate_truth()
    # print(len(infect_rate), len(alldates))


    confirm, remove = read_confirm_data()

    ump_data, last_data = read_ump_truth()
    ump_data = last_data[:6] + ump_data
    # print(len(ump_data), len(last_data))
 
    mobility = read_mobility_truth('training')
    mob7 = mobility[:7, :]
    # print(mobility.shape)

    new = read_confirm_pred(if_norm='test')
    new7 = new[:7]

    last7 = ump_data[:7]
    ump7 = ump_data[:7]
    
    x = torch.FloatTensor([328239523, get_E0(), confirm[0]-remove[0], remove[0]]).cuda()


    m0 = mobility[6]
    u0 = ump_data[6]
    i0 = infect_rate[0]

    out, mob_out, inf_out, true_inf, ump_out = test_lstm(len(alldates)+90, x, m0, u0, i0, control, new7, ump7, mob7, last7, new_mu, new_sigma, inf_mu, inf_sigma, ump_data[7:], mobility[6:], oxc_data, infect_rate)


    daily = list((out[:, 2] + out[:, 3]).cpu().detach().numpy())

    ds = []
    for i in range(1, len(daily)):
        ds.append(daily[i] - daily[i-1])
    dd = []
    for i in range(1, len(confirm)):
        dd.append(confirm[i] - confirm[i-1])

    rem = list((out[:, 3]).cpu().detach().numpy())

    
    # for idx in range(6):
    #     paint(mob_out[:, idx], mobility[7:, idx], '/test/m{}'.format(idx))

    # paint(inf_out, infect_rate[1:], '/test/infection rate')
    # paint(ds, dd, '/test/daily')
    # paint(daily, confirm, '/test/accumulate confirm')
    # paint(rem, remove, '/test/accumulate remove')
    # paint(ump_out, ump_data[7:], '/test/unemployment rate')
    
    # write_mobility(mob_out)
    # write_inf(inf_out)
    # write_confirm(ds)
    # write_unemployment(ump_out)

    for idx in range(6):
        mob_out[:, idx] = mob_out[:, idx] * (max_min_param['m{}'.format(idx)][1]-max_min_param['m{}'.format(idx)][0]) + max_min_param['m{}'.format(idx)][0]

    
    for i in range(len(ump_out)):
        ump_out[i] = ump_out[i] * (max_min_param['ump'][1]-max_min_param['ump'][0]) + max_min_param['ump'][0]
    

    open_str = ''
    # open_str = 'open'
    print(begin_change_date, args.comb, args.idx)
    # csvdir = '../result/csvs_diff_begin_frontier_ave/{}/{}/'.format(begin_change_date, args.comb)
    # csvdir = '../result/csvs_diff_begin_ave/{}/{}/'.format(begin_change_date, args.idx)
    
    csvdir = '../result/not_open_ave/'
    # if not os.path.exists(csvdir):
    #     os.makedirs(csvdir)

    write_csv(mob_out[:, 0], csvdir + open_str + 'm0.csv')
    write_csv(mob_out[:, 1], csvdir + open_str + 'm1.csv')
    write_csv(mob_out[:, 2], csvdir + open_str + 'm2.csv')
    write_csv(mob_out[:, 3], csvdir + open_str + 'm3.csv')
    write_csv(mob_out[:, 4], csvdir + open_str + 'm4.csv')
    write_csv(mob_out[:, 5], csvdir + open_str + 'm5.csv')

    write_csv(ump_out, csvdir + open_str + 'ump.csv')
    write_csv(ds, csvdir + open_str + 'daily.csv')
    write_csv(daily, csvdir + open_str + 'acc confirm.csv')



if __name__ == "__main__":
    main()

