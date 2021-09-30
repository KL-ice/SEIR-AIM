from config import *
from utils import *
from models import *
from data_reader import *


def train_mobility_epoch(train_data, control, model, optimizer, idx):
    _s = 0

    for i in range(train_data['len']):

        _new = torch.FloatTensor(train_data['new'][i]).cuda().view(1, -1)
        _ump = torch.FloatTensor(train_data['ump'][i]).cuda().view(1, -1)
        _oxc = torch.FloatTensor(train_data['oxc'][i]).cuda().view(1, -1)
        _y = torch.FloatTensor([train_data['mob'][i][idx]]).cuda().view(1, -1)

        if idx != 5:
            _oxc = _oxc[:, 1:]

        mob = model(_new, control, _ump, _oxc)

        ump_noise = _ump + torch.rand(1, 7).cuda() * torch.sqrt(0.01*torch.abs(_ump))
        mob_noise_ump = model(_new, control, ump_noise, _oxc)

        _new_noise = _new + torch.rand(1, 7).cuda() * torch.sqrt(0.005*torch.abs(_new))
        mob_noise_new = model(_new_noise, control, _ump, _oxc)

        mob_noise_both = model(_new_noise, control, ump_noise, _oxc)

        optimizer.zero_grad()
        loss = F.mse_loss(mob, _y) + (F.mse_loss(mob_noise_ump, mob) + F.mse_loss(mob_noise_new, mob) + F.mse_loss(mob_noise_both, mob)) / 3
        loss.backward()
        optimizer.step()

        cut_param(model, idx)
        _s += loss.item()

    print(_s)


def test_monbility_epoch(test_data, control, model, idx):

    _s = 0
    _k = 0
    for i in range(test_data['len']):
        _new = torch.FloatTensor(test_data['new'][i]).cuda().view(1, -1)
        _ump = torch.FloatTensor(test_data['ump'][i]).cuda().view(1, -1)
        _oxc = torch.FloatTensor(test_data['oxc'][i]).cuda().view(1, -1)
        _y = torch.FloatTensor([test_data['mob'][i][idx]]).cuda().view(1, -1)

        if idx != 5:
            _oxc = _oxc[:, 1:]

        mob = model(_new, control, _ump, _oxc)

        loss = F.mse_loss(mob, _y)

        _s += loss.item()
        _k += (abs(mob - _y) / (_y)).item()

    print(_s, 1-(_k / test_data['len']))
    return _s


def write_pred_mobility(data, control, models):
    rows = []
    
    csvFile = open('../../row_data/mobility_truth.csv', 'r')
    reader = csv.reader(csvFile)
    for item in reader:
        if reader.line_num == 1:
            continue
        if item[4] < alldates[0]:
            rows.append(item[4:])
    csvFile.close()
    for idx in range(6):
        models[idx].eval()

    for i in range(len(data['oxc'])):
        _new = torch.FloatTensor(data['new'][i]).cuda().view(1, -1)
        _ump = torch.FloatTensor(data['ump'][i]).cuda().view(1, -1)
        _oxc = torch.FloatTensor(data['oxc'][i]).cuda().view(1, -1)

        mobs = []
        for idx in range(6):
            if idx != 5:
                mob = models[idx](_new, control, _ump, _oxc[:, 1:]).item()
            else:
                mob = models[idx](_new, control, _ump, _oxc).item()
            mob = mob * ((max_min_param['m{}'.format(idx)][1] - max_min_param['m{}'.format(idx)][0])) + max_min_param['m{}'.format(idx)][0]
            mobs.append(mob)

        row = [alldates[i]] + mobs

        rows.append(row)

    csvFile = open('../snap_data/mobility_pred.csv', 'w', newline='')
    writer = csv.writer(csvFile)
    writer.writerows(rows)
    
    csvFile.close()


def paint_mobility_pred(data, control, models):

    for idx in range(6):
        models[idx].eval()

        mobs = []
        for i in range(len(data['oxc'])):
            _new = torch.FloatTensor(data['new'][i]).cuda().view(1, -1)
            _ump = torch.FloatTensor(data['ump'][i]).cuda().view(1, -1)
            _oxc = torch.FloatTensor(data['oxc'][i]).cuda().view(1, -1)
            
            if idx != 5:
                mob = models[idx](_new, control, _ump, _oxc[:, 1:]).item()
            else:
                mob = models[idx](_new, control, _ump, _oxc).item()
            mobs.append(mob)

        paint(mobs, data['mob'][:, idx], 'm{}'.format(idx))


def train_mobility(models, optimizers, data, control, train_iter):
    if_all = ''
    if train_iter > 20:
        if_all = '_all'
    fix_random_seed()
    path = '../models/mobility/'
    if not os.path.exists(path):
        os.makedirs(path)

    
    train_data, test_data = split_train_set(data, loss_weight)


    for idx in range(6):
        min_loss = 1000
        count = 0
        for epoch_num in range(100000):
            models[idx].train()
            print(train_iter, 'training mobility idx: {}, epoch:{}'.format(idx, epoch_num))
            if epoch_num % 100 == 0:
                fintune_rate(optimizers[idx], 0.85)
            train_mobility_epoch(train_data, control, models[idx], optimizers[idx], idx)
            models[idx].eval()
            test_loss = test_monbility_epoch(test_data, control, models[idx], idx)

            if test_loss < min_loss:
                min_loss = test_loss
                print('-----------------------------update model in epoch', epoch_num)
                torch.save(models[idx], path + 'm{}{}_iter{}.pkl'.format(idx, if_all, train_iter))
                count = 0
            else:
                count += 1
                if count >= 100:
                    break
    
    write_pred_mobility(data, control, models)
    paint_mobility_pred(data, control, models)


if __name__ == '__main__':
    models, optimizers = [], []
    
    for i in range(6):
        if i != 5:
            mod = m_model().cuda()
        else:
            mod = m5_model().cuda()
        opt = optim.Adam([{'params':mod.parameters(), 'lr': 6e-4, 'weight_decay': 5e-5}])
        models.append(mod)
        optimizers.append(opt)
    
    train_mobility(models, optimizers)

