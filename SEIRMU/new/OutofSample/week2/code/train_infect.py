from config import *
from utils import *
from models import *
from data_reader import *



def train_inf_epoch(train_data, control, model, optimizer):
    _s = 0
    for i in range(train_data['len']):
        _mob = torch.FloatTensor(train_data['mob'][i]).cuda().view(1, 7, -1)
        _y = torch.FloatTensor([train_data['inf'][i]]).cuda().view(1, 1)

        inf = model(_mob, control)

        _mob_noise = _mob + torch.rand(1, 7, 6).cuda() * torch.sqrt(0.005*torch.abs(_mob))
        inf_noise = model(_mob_noise, control)

        optimizer.zero_grad()
        # loss = F.mse_loss(inf, _y) / _y + 0.5*F.mse_loss(inf_noise, inf)
        loss = F.mse_loss(inf, _y) + 0.5*F.mse_loss(inf_noise, inf)
        loss.backward()
        optimizer.step()

        _s += loss.item()
    
    print(_s)


def test_inf_epoch(test_data, control, model):
    _s = 0
    _k = 0
    # for name,parameters in model.named_parameters():
    #     print(name, parameters)
    for i in range(test_data['len']):
        _mob = torch.FloatTensor(test_data['mob'][i]).cuda().view(1, 7, -1)
        _y = torch.FloatTensor([test_data['inf'][i]]).cuda().view(1, 1)

        inf = model(_mob, control)

        loss = F.mse_loss(inf, _y) / _y
        _s += loss.item()
        # print(inf, _y)
        _k += (abs(inf - _y) / (_y)).item()
    
    print(_s, 1-(_k / test_data['len']))
    return _s


def write_pred_inf(data, control, model):
    model.eval()
    rows = []
    
    csvFile = open('../../row_data/E.csv', 'r')
    reader = csv.reader(csvFile)
    for item in reader:
        if item[0] < alldates[0]:
            rows.append(item)
    csvFile.close()

    for i in range(len(data['mob'])):
        _mob = torch.FloatTensor(data['mob'][i]).cuda().view(1, 7, -1)
        inf = model(_mob, control)

        row = [alldates[i], inf.item(), inf.item()]
        rows.append(row)

    csvFile = open('../snap_data/inf_pred.csv', 'w', newline='')
    writer = csv.writer(csvFile)
    writer.writerows(rows)
    
    csvFile.close()


def paint_inf_pred(data, control, model):
    infs = []
    for i in range(len(data['mob'])):
        _mob = torch.FloatTensor(data['mob'][i]).cuda().view(1, 7, -1)
        inf = model(_mob, control).item()
        infs.append(inf)
    
    paint(infs, data['inf'], 'inf')


def train_inf(model, optimizer, data, control, train_iter):
    if_all = ''
    if train_iter > 0:
        if_all = '_all'

    fix_random_seed()
    path = '../models/infection/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    train_data, test_data = split_train_set(data, loss_weight)

    min_loss = 1000
    count = 0

    for i in range(10000):
        model.train()
        print(train_iter, 'training inf epoch:', i)
        if i % 100 == 0:
            fintune_rate(optimizer, 0.6)
        train_inf_epoch(train_data, control, model, optimizer)

        model.eval()
        test_loss = test_inf_epoch(test_data, control, model)

        if test_loss < min_loss:
            min_loss = test_loss
            print('-----------------------------update model in epoch', i)
            torch.save(model, path + 'infect{}_iter{}.pkl'.format(if_all, train_iter))
            count = 0
        else:
            count += 1
            if count >= 100:
                break
    
    model.eval()
    write_pred_inf(data, control, model)
    paint_inf_pred(data, control, model)


if __name__ == '__main__':
    infect_net = infect_nn_lstm_128().cuda()
    infect_optim = optim.Adam([{'params':infect_net.parameters(), 'lr': 6e-4, 'weight_decay':1e-4}])
    
    train_inf(infect_net, infect_optim)
