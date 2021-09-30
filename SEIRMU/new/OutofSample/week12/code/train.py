import os
from config import *
from utils import *
from models import *
from data_reader import *


from train_infect import *
from train_mobility import *
from train_unemployment import *
from train_seir import *


parser = argparse.ArgumentParser()
parser.add_argument('--week', type=int, default=0, help='idx')


def train_all():
    mode = 'self'

    control = read_control()

    for train_iter in range(21, 40):
        
        if train_iter >= 21:
            os.system('python seirsu_test.py --iternum {} {} {} {}'.format(train_iter-1, train_iter-1, train_iter-1, train_iter-1))
        if train_iter >= 21:
            mode = 'all'
        fix_random_seed()
        if train_iter == 0:
            mobility = read_mobility_truth('training')
        else:
            mobility = read_mobility_pred(mode)
        ump_data, last_data = read_ump_truth()
        mobility_data = []
        for i in range(len(mobility)-6):
            mobility_data.append(mobility[i: i+7])
        data = {'mob':mobility_data, 'ump':ump_data, 'last':last_data}
        if train_iter == 0:
            ump_net = unemploy().cuda()
            ump_optim = optim.Adam([{'params':ump_net.parameters(), 'lr': 1e-3, 'weight_decay': 5e-5}])
        elif train_iter > 21:
            ump_net = torch.load('../models/unemployment/ump_model_all_iter{}.pkl'.format(train_iter-1)).train()
            ump_optim = optim.Adam([{'params':ump_net.parameters(), 'lr': 5e-5, 'weight_decay': 5e-5}])
        else:
            ump_net = torch.load('../models/unemployment/ump_model_iter{}.pkl'.format(train_iter-1)).train()
            ump_optim = optim.Adam([{'params':ump_net.parameters(), 'lr': 5e-5, 'weight_decay': 5e-5}])
        train_ump(ump_net, ump_optim, data, control, train_iter)

        if train_iter >= 21:
            os.system('python seirsu_test.py --iternum {} {} {} {}'.format(train_iter, train_iter-1, train_iter-1, train_iter-1))
        fix_random_seed()
        train_seir(train_iter)

        if train_iter >= 21:
            os.system('python seirsu_test.py --iternum {} {} {} {}'.format(train_iter, train_iter, train_iter-1, train_iter-1))
        fix_random_seed()
        new = read_confirm_pred(mode, if_norm='train')
        ump_data= read_unemployment_pred(mode)
        mobility_data = read_mobility_truth('truth')
        oxc_data = read_oxc_onehot()
        _x = []
        _u = []
        for i in range(7, len(new)):
            _x.append(new[i-7:i])
            _u.append(ump_data[i-7:i])
        print(len(new))
        data = {'new':_x, 'ump':_u, 'oxc':oxc_data, 'mob':mobility_data}
        models, optimizers = [], []
        for idx in range(6):
            if train_iter == 0:
                if idx != 5:
                    mod = m_model().cuda()
                else:
                    mod = m5_model().cuda()
                opt = optim.Adam([{'params':mod.parameters(), 'lr': 6e-4, 'weight_decay': 5e-5}])
            elif train_iter > 21:
                mod = torch.load('../models/mobility/m{}_all_iter{}.pkl'.format(idx, train_iter-1)).train()
                opt = optim.Adam([{'params':mod.parameters(), 'lr': 1e-5, 'weight_decay': 5e-5}])
            else:
                mod = torch.load('../models/mobility/m{}_iter{}.pkl'.format(idx, train_iter-1)).train()
                opt = optim.Adam([{'params':mod.parameters(), 'lr': 1e-5, 'weight_decay': 5e-5}])
            models.append(mod)
            optimizers.append(opt)
        train_mobility(models, optimizers, data, control, train_iter)


        if train_iter >= 21:
            os.system('python seirsu_test.py --iternum {} {} {} {}'.format(train_iter, train_iter, train_iter, train_iter-1))
        if train_iter >= 21:
            mode = 'all'
        fix_random_seed()
        mobility = read_mobility_pred(mode)
        infect_rate = read_infect_rate_truth()
        _mob_data = []
        for i in range(7, len(mobility)+1):
            _mob_data.append(mobility[i-7:i])
        data = {'mob':_mob_data, 'inf':infect_rate}
        if train_iter == 0:
            infect_net = infect_nn_lstm_128().cuda()
            infect_optim = optim.Adam([{'params':infect_net.parameters(), 'lr': 6e-4, 'weight_decay':5e-5}])
        elif train_iter > 21:
            infect_net = torch.load('../models/infection/infect_all_iter{}.pkl'.format(train_iter-1)).train()
            infect_optim = optim.Adam([{'params':infect_net.parameters(), 'lr': 1e-6, 'weight_decay':5e-5}])
        else:
            infect_net = torch.load('../models/infection/infect_iter{}.pkl'.format(train_iter-1)).train()
            infect_optim = optim.Adam([{'params':infect_net.parameters(), 'lr': 1e-6, 'weight_decay':5e-5}])
        train_inf(infect_net, infect_optim, data, control, train_iter)

    return


if __name__ == "__main__":
    train_all()