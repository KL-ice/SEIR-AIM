from config import *
from utils import *
from models import *
from data_reader import *


def train_unemployment_epoch(train_data, control, model, optimizer):
    _s = 0
    for i in range(train_data['len']):
        _mob = torch.FloatTensor(train_data['mob'][i]).cuda().view(1, 7, -1)
        _last = torch.FloatTensor([train_data['last'][i]]).cuda().view(1, 1)
        _y = torch.FloatTensor([train_data['ump'][i]]).cuda().view(1, 1)


        ump = model(_mob, control, _last)

        _mob_noise = _mob + torch.rand(1, 7, 1).cuda() * torch.sqrt(0.01*_mob)
        _ump_noise = model(_mob_noise, control, _last)

        optimizer.zero_grad()
        loss = F.mse_loss(ump, _y)# + F.mse_loss(_ump_noise, ump)
        loss.backward()
        optimizer.step()

        _s += loss.item()
    print(_s)


def test_unemployment_epoch(test_data, control, model):
    _s = 0
    _k = 0
    for i in range(test_data['len']):

        _mob = torch.FloatTensor(test_data['mob'][i]).cuda().view(1, 7, -1)
        _last = torch.FloatTensor([test_data['last'][i]]).cuda().view(1, 1)
        _y = torch.FloatTensor([test_data['ump'][i]]).cuda().view(1, 1)

        ump = model(_mob, control, _last)

        loss = F.mse_loss(ump, _y)
        _s += loss.item()
        _k += (abs(ump - _y) / (_y)).item()
    
    print(_s, 1-(_k / test_data['len']))
    return _s


def write_pred_ump(data, control, model):
    model.eval()
    # for name,parameters in model.named_parameters():
    #     print(name, parameters)
    rows = []
    
    csvFile = open('../../row_data/unemploy.csv', 'r')
    reader = csv.reader(csvFile)
    for item in reader:
        if item[0] < alldates[0]:
            rows.append(item)
    csvFile.close()

    for i in range(len(data['mob'])):
        _mob = torch.FloatTensor(data['mob'][i]).cuda().view(1, 7, -1)
        _last = torch.FloatTensor([data['last'][i]]).cuda().view(1, 1)

        ump = model(_mob, control, _last).item()
        ump = ump * (max_min_param['ump'][1] - max_min_param['ump'][0]) + max_min_param['ump'][0]
        rows.append([alldates[i], ump])

    csvFile = open('../snap_data/ump_pred.csv', 'w', newline='')
    writer = csv.writer(csvFile)
    writer.writerows(rows)
    csvFile.close()


def paint_ump_pred(data, control, model):
    model.eval()
    umps = []
    for i in range(len(data['mob'])):
        _mob = torch.FloatTensor(data['mob'][i]).cuda().view(1, 7, -1)
        _last = torch.FloatTensor([data['last'][i]]).cuda().view(1, 1)

        ump = model(_mob, control, _last).item()
        # print(_mob, ump, _last)
        umps.append(ump)
    
    paint(umps, data['ump'], 'ump')


def train_ump(model, optimizer, data, control, train_iter):
    if_all = ''
    if train_iter > 50:
        if_all = '_all'
    fix_random_seed()
    path = '../models/unemployment/'
    if not os.path.exists(path):
        os.makedirs(path)

    train_data, test_data = split_train_set(data, loss_weight)

    min_loss = 1000
    count = 0

    for i in range(10000):
        model.train()
        print(train_iter, 'training ump epoch:', i)
        if i % 200 == 0:
            fintune_rate(optimizer, 0.8)
        train_unemployment_epoch(train_data, control, model, optimizer)
        model.eval()
        test_loss = test_unemployment_epoch(test_data, control, model)

        if test_loss < min_loss:
            min_loss = test_loss
            print('-----------------------------update model in epoch', i)
            torch.save(model, path + 'ump_model{}_iter{}.pkl'.format(if_all, train_iter))
            count = 0
        else:
            count += 1
            if count >= 100:
                break
    
    write_pred_ump(data, control, model)
    paint_ump_pred(data, control, model)


if __name__ == "__main__":
    ump_model = unemploy().cuda()
    ump_optim = optim.Adam([{'params':ump_model.parameters(), 'lr': 1e-3, 'weight_decay': 5e-5}])
    train_ump(ump_model, ump_optim)
    