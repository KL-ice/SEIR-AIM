import torch
import torch.nn as nn
import torch.nn.functional as F


class m5_model(nn.Module):
    def __init__(self):
        super(m5_model, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.lstm_ump = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)

        self.cc = nn.Linear(in_features=5, out_features=3, bias=True)

        self.policy_c = torch.nn.Parameter(torch.FloatTensor(8, 1), requires_grad=True)
        self.policy_others = torch.nn.Parameter(torch.FloatTensor(6, 1), requires_grad=True)
        self.B = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)

        torch.nn.init.xavier_uniform(self.policy_c.data, gain=1.414)
        torch.nn.init.xavier_uniform(self.policy_others.data, gain=1.414)
        torch.nn.init.xavier_uniform(self.B.data, gain=1.414)

        self.policy_c.data = torch.where(self.policy_c.data<0, self.policy_c.data*-1, self.policy_c.data)
        self.policy_c.data = torch.where(self.policy_c.data==0, torch.ones(8, 1)*0.5, self.policy_c.data)
        # self.policy_c.data = torch.ones(8, 1)*0.1

        self.blm_param = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.blm_param.data = torch.ones(1, 1) * -0.1

        self.B2 = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        torch.nn.init.xavier_uniform(self.B2.data, gain=1.414)
        

        self.fc = nn.Sequential(nn.Linear(in_features=32+3+16 + 1 + 1, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=1, bias=True))
        
        self.relu = nn.ReLU()


    def forward(self, x, controls, ump7, policy):
        b, _ = x.shape

        mobility = x.view(b, 1, 7).permute(0, 2, 1)
        control = self.cc(controls)
        o, (mobility, c) = self.lstm1(mobility)
        mobility = mobility.view(b, -1)

        ump7 = ump7.view(b, 1, 7).permute(0, 2, 1)
        o, (ump7, c) = self.lstm_ump(ump7)
        ump7 = ump7.view(b, -1)

        param = torch.cat((self.policy_c, self.policy_others), dim=0)

        oxc = torch.matmul(policy[:, 1:], param) + self.B

        blm = torch.matmul(policy[:, :1], self.blm_param) + self.B2

        x = torch.cat((mobility, control, ump7, oxc, blm), dim=1)
        output = self.fc(x)

        return output


class m4_model(nn.Module):
    def __init__(self):
        super(m4_model, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.lstm_ump = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)

        self.cc = nn.Linear(in_features=5, out_features=3, bias=True)

        self.policy_c = torch.nn.Parameter(torch.FloatTensor(8, 1), requires_grad=True)
        self.policy_others = torch.nn.Parameter(torch.FloatTensor(5, 1), requires_grad=True)
        
        # 初始化方法

        self.fc = nn.Sequential(nn.Linear(in_features=32+3+16+1, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=1, bias=True))
        
        self.relu = nn.ReLU()


    def forward(self, x, controls, ump7, policy):
        b, _ = x.shape

        mobility = x.view(b, 1, 7).permute(0, 2, 1)
        control = self.cc(controls)
        o, (mobility, c) = self.lstm1(mobility)
        mobility = mobility.view(b, -1)

        ump7 = ump7.view(b, 1, 7).permute(0, 2, 1)
        o, (ump7, c) = self.lstm_ump(ump7)
        ump7 = ump7.view(b, -1)

        param = torch.cat((self.relu(self.policy_c), self.policy_others), dim=0)
        # print(param)


        oxc = torch.matmul(policy, param)
        # print(oxc)

        x = torch.cat((mobility, control, ump7, oxc), dim=1)
        output = self.fc(x)

        return output


class m_model(nn.Module):
    def __init__(self):
        super(m_model, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.lstm_ump = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)

        self.cc = nn.Linear(in_features=5, out_features=3, bias=True)

        self.policy_c = torch.nn.Parameter(torch.FloatTensor(8, 1), requires_grad=True)
        self.policy_others = torch.nn.Parameter(torch.FloatTensor(6, 1), requires_grad=True)

        torch.nn.init.xavier_uniform(self.policy_c.data, gain=1.414)
        torch.nn.init.xavier_uniform(self.policy_others.data, gain=1.414)

        self.policy_c.data = torch.where(self.policy_c.data>0, self.policy_c.data*-1, self.policy_c.data)
        self.policy_c.data = torch.where(self.policy_c.data==0, torch.ones(8, 1)*-0.5, self.policy_c.data)


        self.fc = nn.Sequential(nn.Linear(in_features=32+3+16 + 1, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=1, bias=True))


    def forward(self, x, controls, ump7, policy):
        b, _ = x.shape

        mobility = x.view(b, 1, 7).permute(0, 2, 1)
        control = self.cc(controls)
        o, (mobility, c) = self.lstm1(mobility)
        mobility = mobility.view(b, -1)

        ump7 = ump7.view(b, 1, 7).permute(0, 2, 1)
        o, (ump7, c) = self.lstm_ump(ump7)
        ump7 = ump7.view(b, -1)

        param = torch.cat((self.policy_c, self.policy_others), dim=0)
        # print(param)

        oxc = torch.matmul(policy, param)
        # print(oxc)
        x = torch.cat((mobility, control, ump7, oxc), dim=1)
        # print(self.lstm_ump.weight)

        output = self.fc(x)
        # print(x)
        return output


class m_model32(nn.Module):
    def __init__(self):
        super(m_model32, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.lstm_ump = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)

        self.cc = nn.Linear(in_features=5, out_features=3, bias=True)

        self.policy_c = torch.nn.Parameter(torch.FloatTensor(8, 1), requires_grad=True)
        self.policy_others = torch.nn.Parameter(torch.FloatTensor(5, 1), requires_grad=True)

        torch.nn.init.xavier_uniform(self.policy_c.data, gain=1.414)
        torch.nn.init.xavier_uniform(self.policy_others.data, gain=1.414)

        self.policy_c.data = torch.where(self.policy_c.data>0, self.policy_c.data*-1, self.policy_c.data)
        self.policy_c.data = torch.where(self.policy_c.data==0, torch.ones(8, 1)*-0.5, self.policy_c.data)


        self.fc = nn.Sequential(nn.Linear(in_features=32+3+32 + 1, out_features=32, bias=True),
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=32, out_features=16, bias=True), 
                                nn.Tanh(),
                                nn.Linear(in_features=16, out_features=1, bias=True))


    def forward(self, x, controls, ump7, policy):
        b, _ = x.shape

        mobility = x.view(b, 1, 7).permute(0, 2, 1)
        control = self.cc(controls)
        o, (mobility, c) = self.lstm1(mobility)
        mobility = mobility.view(b, -1)

        ump7 = ump7.view(b, 1, 7).permute(0, 2, 1)
        o, (ump7, c) = self.lstm_ump(ump7)
        ump7 = ump7.view(b, -1)

        param = torch.cat((self.policy_c, self.policy_others), dim=0)
        # print(param)

        oxc = torch.matmul(policy, param)
        # print(oxc)
        x = torch.cat((mobility, control, ump7, oxc), dim=1)
        # print(self.lstm_ump.weight)

        output = self.fc(x)
        # print(x)
        return output


class m_model_fc(nn.Module):
    def __init__(self):
        super(m_model_fc, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.lstm_ump = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)

        self.cc = nn.Linear(in_features=5, out_features=3, bias=True)
        self.policy_fc = nn.Sequential(nn.Linear(in_features=13, out_features=4, bias=False))

        self.fc = nn.Sequential(nn.Linear(in_features=32+3+16+4, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=1, bias=True))


    def forward(self, x, controls, ump7, policy):
        b, _ = x.shape

        mobility = x.view(b, 1, 7).permute(0, 2, 1)
        control = self.cc(controls)
        o, (mobility, c) = self.lstm1(mobility)
        mobility = mobility.view(b, -1)

        ump7 = ump7.view(b, 1, 7).permute(0, 2, 1)
        o, (ump7, c) = self.lstm_ump(ump7)
        ump7 = ump7.view(b, -1)

        oxc = self.policy_fc(policy)

        x = torch.cat((mobility, control, ump7, oxc), dim=1)
        output = self.fc(x)

        return output


class m_model_idx2(nn.Module):
    def __init__(self):
        super(m_model_idx2, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=32, batch_first=True, bidirectional=True)
        self.lstm_ump = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)

        self.cc = nn.Linear(in_features=5, out_features=3, bias=True)
        self.fc = nn.Sequential(nn.Linear(in_features=32+3+16, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=1, bias=True))


    def forward(self, x, controls, ump7):
        b, _ = x.shape

        mobility = x.view(b, 1, 7).permute(0, 2, 1)
        control = self.cc(controls)
        o, (mobility, c) = self.lstm1(mobility)
        mobility = mobility.view(b, -1)

        ump7 = ump7.view(b, 1, 7).permute(0, 2, 1)
        o, (ump7, c) = self.lstm_ump(ump7)
        ump7 = ump7.view(b, -1)

        x = torch.cat((mobility, control, ump7), dim=1)
        output = self.fc(x)

        return output


class m_model_linear(nn.Module):
    def __init__(self):
        super(m_model_linear, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.lstm_ump = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)

        self.inf_fc = nn.Sequential(nn.Linear(in_features=7, out_features=10, bias=True), 
                                    nn.Tanh())
        self.ump_fc = nn.Sequential(nn.Linear(in_features=7, out_features=10, bias=True), 
                                    nn.Tanh())

        self.cc = nn.Linear(in_features=6, out_features=3, bias=True)
        self.fc = nn.Sequential(nn.Linear(in_features=23, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=1, bias=True))

        self.out_layer = nn.Linear(in_features=2, out_features=1, bias=True)

    def forward(self, x, controls, last, ump7):
        b, _ = x.shape

        inf = x.view(b, 7)
        inf = self.inf_fc(inf)

        ump7 = ump7.view(b, 7)
        ump7 = self.inf_fc(ump7)

        control = self.cc(controls)

        x = torch.cat((inf, control, ump7), dim=1)
        output = self.fc(x)

        # output = torch.cat([output, last], dim=1)
        # output = self.out_layer(output)

        return output


class m_model_lstm(nn.Module):
    def __init__(self):
        super(m_model_lstm, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=16, batch_first=True, bidirectional=True)
        self.lstm_ump = nn.LSTM(input_size=1, hidden_size=8, batch_first=True, bidirectional=True)

        self.cc = nn.Linear(in_features=5, out_features=3, bias=True)
        self.fc = nn.Sequential(nn.Linear(in_features=32+3+16, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=1, bias=True))


    def forward(self, x, controls, ump7):
        b, _ = x.shape
        control = self.cc(controls)

        mobility = x.view(b, 1, 7).permute(0, 2, 1)
        ump7 = ump7.view(b, 1, 7).permute(0, 2, 1)

        # mobility = torch.cat([mobility, ump7], dim=-1)

        o, (mobility, c) = self.lstm1(mobility)
        mobility = mobility.permute(1, 0, 2).reshape(b, -1)

        o, (ump7, c) = self.lstm_ump(ump7)
        ump7 = ump7.permute(1, 0, 2).reshape(b, -1)


        x = torch.cat((mobility, control, ump7), dim=1)
        output = self.fc(x)

        return output


class m0_model(nn.Module):
    def __init__(self):
        super(m0_model, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.lstm_ump = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)

        # self.dropout = nn.Dropout(0.6)
        self.cc = nn.Linear(in_features=6, out_features=3, bias=True)
        self.fc = nn.Sequential(nn.Linear(in_features=32+3+16, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=1, bias=True),
                                )#nn.Tanh())

    def forward(self, x, controls, last, ump7):
        b, _ = x.shape

        mobility = x.view(b, 1, 7).permute(0, 2, 1)
        control = self.cc(controls)

        ump7 = ump7.view(b, 1, 7).permute(0, 2, 1)
        o, (ump7, c) = self.lstm_ump(ump7)

        o, (mobility, c) = self.lstm1(mobility)

        mobility = mobility.view(b, -1)
        ump7 = ump7.view(b, -1)

        x = torch.cat((mobility, control, ump7), dim=1)

        output = self.fc(x)
        return output


class infect_nn(nn.Module):
    def __init__(self):
        super(infect_nn, self).__init__()
        
        self.cc = nn.Linear(in_features=5, out_features=3)
        self.mc = nn.Sequential(nn.Linear(in_features=6, out_features=10),
                                nn.Tanh())
        self.fc = nn.Sequential(nn.Linear(in_features=13, out_features=32, bias=True), 
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=32, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=16, out_features=1, bias=True))
        
        self.output_layer = nn.Linear(in_features=2, out_features=1, bias=True)
        # for i in [0, 3, 5]:
        #     torch.nn.init.xavier_uniform_(self.fc[i].weight)

    def forward(self, mobility, controls, last):
        control = self.cc(controls)
        m = self.mc(mobility)
        x = torch.cat((m, control), dim=1)

        # print(x)
        output = self.fc(x)

        output = torch.cat([output, last], dim=1)
        output = self.output_layer(output)

        return output


class SEIR(nn.Module):
    def __init__(self, a, b, c):
        super(SEIR, self).__init__()
        
        self.alpha = nn.Parameter(torch.tensor([a])) # 2.04193505e-02
        self.beta = nn.Parameter(torch.tensor([b])) # 3.58661213e-07
        self.gamma = nn.Parameter(torch.tensor([c])) # 2e-2

    def forward(self, x, infect_rate):

        self.alpha.data = F.relu(self.alpha)
        self.beta.data = F.relu(self.beta)
        self.gamma.data = F.relu(self.gamma)

        Y = torch.zeros(4).cuda()
        Y[0] = -infect_rate * self.beta * x[0] * x[2]
        Y[1] = infect_rate * self.beta * x[0] * x[2] - self.alpha * x[1]
        Y[2] = self.alpha * x[1] - self.gamma * x[2]
        Y[3] = self.gamma * x[2]

        return Y + x


class infect_nn_social(nn.Module):
    def __init__(self):
        super(infect_nn_social, self).__init__()
        
        self.cc = nn.Linear(in_features=6, out_features=3)
        self.mc = nn.Sequential(nn.Linear(in_features=6, out_features=10),
                                nn.Tanh())
        self.fc = nn.Sequential(nn.Linear(in_features=13, out_features=32, bias=True), 
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=32, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=16, out_features=1, bias=True))
        
        self.output_layer = nn.Linear(in_features=2, out_features=1, bias=True)
        # for i in [0, 3, 5]:
        #     torch.nn.init.xavier_uniform_(self.fc[i].weight)

    def forward(self, mobility, controls, last):

        control = self.cc(controls)
        m = self.mc(mobility)
        x = torch.cat((m, control), dim=1)

        # print(x)
        output = self.fc(x)

        output = torch.cat([output, last], dim=1)
        output = self.output_layer(output)

        return output


class infect_nn_truth(nn.Module):
    def __init__(self):
        super(infect_nn_truth, self).__init__()
        
        self.cc = nn.Linear(in_features=6, out_features=3)
        self.mc = nn.Sequential(nn.Linear(in_features=7, out_features=10),
                                nn.Tanh())
        self.fc = nn.Sequential(nn.Linear(in_features=13, out_features=32, bias=True), 
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=32, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=16, out_features=1, bias=True))
        
        self.output_layer = nn.Linear(in_features=2, out_features=1, bias=True)
        # for i in [0, 3, 5]:
        #     torch.nn.init.xavier_uniform_(self.fc[i].weight)

    def forward(self, mobility, controls, last):

        mobility = torch.cat([mobility, controls[:, -1:]], dim=1)

        control = self.cc(controls)

        m = self.mc(mobility)
        x = torch.cat((m, control), dim=1)

        # print(x)
        output = self.fc(x)

        output = torch.cat([output, last], dim=1)
        output = self.output_layer(output)

        return output


class infect_nn_lstm(nn.Module):
    def __init__(self):
        super(infect_nn_lstm, self).__init__()
        
        self.cc = nn.Linear(in_features=5, out_features=3)
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)

        self.fc = nn.Sequential(nn.Linear(in_features=16 + 3, out_features=16, bias=True), 
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=16, out_features=1, bias=True),
                                nn.ReLU())
        
        self.out_layer = nn.Linear(in_features=2, out_features=1, bias=True)
        

    def forward(self, mobility, controls, last):
        control = self.cc(controls)
        # mobility[:, 6, :2] = mobility[:, 6, :2] + 0.05 * mobility[:, 6, :2]
        # mobility[:, 6, 2] = mobility[:, 6, 2] - 0.05 * mobility[:, 6, 2]

        b, _, _ = mobility.shape
        mobility = mobility.view(b, 7, -1)
        # print(mobility.shape)
        o, (mobility, c) = self.lstm1(mobility)
        m = mobility.view(b, -1)

        x = torch.cat((m, control), dim=1)

        output = self.fc(x)

        # output = torch.cat((output, last), dim=1)
        # output = self.out_layer(output)

        return output


class infect_nn_lstm_64(nn.Module):
    def __init__(self):
        super(infect_nn_lstm_64, self).__init__()
        
        self.cc = nn.Linear(in_features=5, out_features=3)
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)

        self.fc = nn.Sequential(nn.Linear(in_features=64 + 3, out_features=32, bias=True), 
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=32, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=16, out_features=1, bias=True),
                                nn.ReLU())
        
        self.out_layer = nn.Linear(in_features=2, out_features=1, bias=True)
        

    def forward(self, mobility, controls, last):
        control = self.cc(controls)
        # mobility[:, 6, :2] = mobility[:, 6, :2] + 0.05 * mobility[:, 6, :2]
        # mobility[:, 6, 2] = mobility[:, 6, 2] - 0.05 * mobility[:, 6, 2]

        b, _, _ = mobility.shape
        mobility = mobility.view(b, 7, -1)
        # print(mobility.shape)
        o, (mobility, c) = self.lstm1(mobility)
        m = mobility.view(b, -1)

        x = torch.cat((m, control), dim=1)

        output = self.fc(x)

        # output = torch.cat((output, last), dim=1)
        # output = self.out_layer(output)

        return output


class infect_nn_lstm_128(nn.Module):
    def __init__(self):
        super(infect_nn_lstm_128, self).__init__()
        
        self.cc = nn.Linear(in_features=5, out_features=3)
        
        self.lstm1 = nn.LSTM(input_size=6, hidden_size=64, batch_first=True)

        self.fc = nn.Sequential(nn.Linear(in_features=64 + 3, out_features=32, bias=True), 
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=32, out_features=16, bias=True),
                                nn.Tanh(),
                                # nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=1, bias=True),
                                nn.ReLU())
        
        self.out_layer = nn.Linear(in_features=2, out_features=1, bias=True)
        

    def forward(self, mobility, controls):
        control = self.cc(controls)
        # mobility[:, 6, :2] = mobility[:, 6, :2] + 0.05 * mobility[:, 6, :2]
        # mobility[:, 6, 2] = mobility[:, 6, 2] - 0.05 * mobility[:, 6, 2]

        b, _, _ = mobility.shape
        mobility = mobility.view(b, 7, -1)
        # print(mobility.shape)
        o, (mobility, c) = self.lstm1(mobility)
        m = mobility.view(b, -1)

        x = torch.cat((m, control), dim=1)

        output = self.fc(x)

        # output = torch.cat((output, last), dim=1)
        # output = self.out_layer(output)

        return output




class infect_nn_lstm_att(nn.Module):
    def __init__(self):
        super(infect_nn_lstm_att, self).__init__()
        
        self.cc = nn.Linear(in_features=5, out_features=3)
        
        self.lstm1 = nn.LSTM(input_size=5, hidden_size=16, batch_first=True)

        self.fc = nn.Sequential(nn.Linear(in_features=16 + 3, out_features=16, bias=True), 
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=16, out_features=1, bias=True),
                                nn.ReLU())
        
        self.Q_weight = nn.Linear(in_features=3, out_features=5, bias=False)
        self.V_weight = nn.Linear(in_features=3, out_features=5, bias=False)


    def forward(self, mobility, controls, last):
        control = self.cc(controls)

        b, _, _ = mobility.shape
        mobility = mobility.view(b, 7, -1)

        Q_value = self.Q_weight(mobility)
        V_value = self.V_weight(mobility)
        mob_att = torch.matmul(torch.matmul(Q_value, Q_value.permute(0, 2, 1)), V_value)

        o, (mobility, c) = self.lstm1(mob_att)
        m = mobility.view(b, -1)

        x = torch.cat((m, control), dim=1)

        output = self.fc(x)

        return output


class infect_nn_lstm_ar(nn.Module):
    def __init__(self):
        super(infect_nn_lstm_ar, self).__init__()
        
        self.cc = nn.Linear(in_features=5, out_features=3)
        
        # self.lstm1 = nn.LSTM(input_size=3, hidden_size=16, batch_first=True)
        self.mc_34 = nn.Linear(in_features=14, out_features=10, bias=True)
        self.mc_5 = nn.Linear(in_features=7, out_features=5, bias=True)

        self.fc = nn.Sequential(nn.Linear(in_features=15 + 3, out_features=16, bias=True), 
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=16, out_features=1, bias=True),
                                nn.ReLU())
        
        self.out_layer = nn.Linear(in_features=2, out_features=1, bias=True)
        

    def forward(self, mobility, controls, last):
        control = self.cc(controls)


        # m = mobility.view(1, -1)
        # m = self.mc(m)

        m34 = mobility[:, :, :2].reshape(1, -1)
        m5 = mobility[:, :, 2:].reshape(1, -1)
        m34 = self.mc_34(m34)
        m5 = self.mc_5(m5)

        print(m34, m5)

        x = torch.cat((m34, m5, control), dim=1)

        output = self.fc(x)


        return output



class infect_nn_cases(nn.Module):
    def __init__(self):
        super(infect_nn_cases, self).__init__()
        self.cc = nn.Linear(in_features=5, out_features=3)
        self.lstm1 = nn.LSTM(input_size=3, hidden_size=16, batch_first=True)
        self.lstm_cases = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(in_features=16 + 3 + 16, out_features=16, bias=True), 
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=16, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=16, out_features=1, bias=True),
                                nn.ReLU())
        self.out_layer = nn.Linear(in_features=2, out_features=1, bias=True)

    def forward(self, mobility, controls, last, cases):
        control = self.cc(controls)
        b, _, _ = mobility.shape
        mobility = mobility.view(b, 7, -1)
        o, (mobility, c) = self.lstm1(mobility)
        m = mobility.view(b, -1)

        cases = cases.view(b, 1, 7).permute(0, 2, 1)
        o, (cases, c) = self.lstm_cases(cases)
        cases = cases.view(b, -1)

        x = torch.cat((m, control, cases), dim=1)

        output = self.fc(x)

        # output = torch.cat((output, last), dim=1)
        # output = self.out_layer(output)

        return output




class unemploy(nn.Module):
    def __init__(self):
        super(unemploy, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=6, hidden_size=64, batch_first=True)
        # self.lstm_inf = nn.LSTM(input_size=2, hidden_size=16, batch_first=True)

        self.cc = nn.Linear(in_features=5, out_features=3)
        self.fc = nn.Sequential(nn.Linear(in_features=64+3, out_features=32, bias=True), 
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=32, out_features=16, bias=True),
                                nn.Tanh(),
                                nn.Linear(in_features=16, out_features=1, bias=True))
        
        # self.out_layer = nn.Sequential(nn.Linear(in_features=2, out_features=1, bias=True),
        #                                nn.ReLU())

        self.out_layer = nn.Linear(in_features=2, out_features=1, bias=True)                               

    def forward(self, x, control, last):
        b, _, _ = x.shape
        mobility = x.view(b, 7, -1)
        control = self.cc(control)

        o, (mobility, c) = self.lstm1(mobility)
        # o, mobility = self.lstm1(mobility)

        mobility = mobility.view(b, -1)

        x = torch.cat((mobility, control), dim=1)
        
        output = self.fc(x)
        # output = torch.cat((output, last), dim=1)
        # output = self.out_layer(output)
        return output
