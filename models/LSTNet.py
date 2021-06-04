import torch
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.P = args.window    ### 目前设置是7
        self.m = data.m
        self.hidR = args.hidRNN   #默认 50
        self.hidC = args.hidCNN   #默认 50
        self.hidS = args.hidSkip  #默认 10
        self.Ck = args.CNN_kernel  # 默认 6
        self.skip = args.skip      # 默认 24
        self.hw = args.highway_window  #默认 24

        # 卷积：输入通道数为1，输出通道数（即卷积核的个数）50，卷积核尺寸(6,228）
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))   # 核大小（6,228）
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)  ## 默认0.2

        if (self.skip > 0):

            #print(self.P,self.Ck,self.skip)
            self.pt = (self.P - self.Ck) / self.skip   #(7-6)/2
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)

        self.output = None

        if (args.output_fun == 'sigmoid'):
            self.output = torch.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = torch.tanh

    def forward(self, x):
        batch_size = x.size(0)
        #print("batch=",batch_size)
        #print("x.shape=",x.shape)  #[1,7,228]

        # CNN
        c = x.view(-1, 1, self.P, self.m)  ## p=24,m=228
        #print('m=', self.m)
        #print("c.shape1=", c.shape)  #[1,1,7,228]
        c = F.relu(self.conv1(c))   ##卷积操作
        #print("c.shape2=", c.shape)  #[1,50,2,1]
        c = self.dropout(c)
        #print("c.shape3=", c.shape)  #[1,50,2,1]
        c = torch.squeeze(c, 3)   ## 对数据维度进行压缩
        #print("c.shape4=", c.shape)  #[1,50,2]

        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        #print("r.shape=", r.shape)
        _, r = self.GRU1(r)

        r = self.dropout(torch.squeeze(r, 0))


        # skip-rnn

        if (self.skip > 0):
            #print('pt=', self.pt)
            self.pt = int(self.pt)
            #print('pt=', self.pt)
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            #print("_____s.shape=", s.shape)   #[1,50,2]

            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res

