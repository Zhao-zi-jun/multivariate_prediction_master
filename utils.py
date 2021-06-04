import torch
import numpy as np
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    # 设置比例：训练集0.6、验证集0.2、测试集0.2
    def __init__(self, file_name, train, valid, device, args):
        self.device = device
        self.P = args.window
        self.h = args.horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        print('rawdat==', self.rawdat.shape)   # (572, 500)

        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape     ## n=572,m=500
        self.scale = np.ones(self.m)
        self._normalized(args.normalize)   ##默认值为2，把数据映射到[0,1]区间。
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.as_tensor(self.scale, device=device, dtype=torch.float)

        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
        fin.close()

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)   ## X=[333, 10, 500] Y=[333, 500]   ## X=[322, 10, 500] Y=[322, 500]
        self.valid = self._batchify(valid_set, self.h)    # X=[114, 10, 500] Y=[114, 500]    # X=[114, 10, 500] Y=[114, 500]
        self.test = self._batchify(test_set, self.h)      #  X=[115, 10, 500] Y=[115, 500]     #X=[115, 10, 500] Y=[115, 500]
                                                          # horizon=1                          # horizon=10

    def _batchify(self, idx_set, horizon):

        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m), device=self.device)
        Y = torch.zeros((n, self.m), device=self.device)

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.as_tensor(self.dat[start:end, :], device=self.device)
            Y[i, :] = torch.as_tensor(self.dat[idx_set[i], :], device=self.device)
        #print(idx_set)
        #print('X.shape==', X.shape)
        #print('Y.shape==', Y.shape)
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length,device=self.device)
        else:
            index = torch.as_tensor(range(length),device=self.device,dtype=torch.long)
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]

            yield Variable(X), Variable(Y)
            start_idx += batch_size
