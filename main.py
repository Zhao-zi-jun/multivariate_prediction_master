import argparse
import math
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models import LSTNet,MHA_Net,CNN,RNN
import importlib
import matplotlib.pyplot as plt

from utils import *
from train_eval import train_method, evaluate_method, makeOptimizer, plot_method

#创建一个解析对象
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
#向对象中添加相关命令行参数或选项,每一个add_argument方法对应一个参数或选项
parser.add_argument('--data', type=str, required=True,help='location of the data file')

parser.add_argument('--model', type=str, default='LSTNet', help='')  #设置模型，修改为model下面的.py文件的名字
parser.add_argument('--window', type=int, default=24 * 7, help='window size')  #窗口大小
parser.add_argument('--horizon', type=int, default=12)

parser.add_argument('--hidRNN', type=int, default=50, help='number of RNN hidden units each layer')##默认层数50
parser.add_argument('--rnn_layers', type=int, default=1, help='number of RNN hidden layers')

parser.add_argument('--hidCNN', type=int, default=50, help='number of CNN hidden units (channels)')##默认层数50
parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=24, help='The window size of the highway component')

parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)

parser.add_argument('--clip', type=float, default=10., help='gradient clipping')   #梯度裁剪
parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit')   # 训练次数上限
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')  #默认批量批量大小32
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)') #忽略20%的特征检测器（让一半的隐层节点值为0），防止过拟合
parser.add_argument('--seed', type=int, default=54321, help='random seed')  #随机种子
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt', help='path to save the final model')

parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--amsgrad', type=str, default=True)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--skip', type=float, default=2)   ## 注意：是浮点型的数字
parser.add_argument('--hidSkip', type=int, default=10)  ##默认为10
parser.add_argument('--L1Loss', type=bool, default=False)  # 把True修改为False ,目的是使用MSE作为损失函数
parser.add_argument('--normalize', type=int, default=0)
parser.add_argument('--output_fun', type=str, default='sigmoid')  #不使用sigmoid
#调用parse_args()方法进行解析
args = parser.parse_args()



# Choose device: cpu or gpu
args.cuda = torch.cuda.is_available()   # 记录当前CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed manually for reproducibility 设置随机种子，确保实验的可重复性
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# 设置比例：训练集0.6、验证集0.2、测试集0.2

Data = Data_utility(args.data, 0.6, 0.2, device, args)

# loss function
if args.L1Loss:
    criterion = nn.L1Loss(reduction='sum')  # MAE：取预测值和真实值的绝对误差的平均数
else:
    criterion = nn.MSELoss(reduction='sum')  #MSE: 预测值和真实值之间的平方和的平均数。
evaluateL2 = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()

# Select model 设置模型
model = eval(args.model).Model(args, Data)

#train_method = train  #方法重命名了，不用管
#eval_method = evaluate

nParams = sum([p.nelement() for p in model.parameters()])  #参数数量
print('number of parameters: %d' % nParams)
if args.cuda:
    model = nn.DataParallel(model)   # DataParallel函数：用多个GPU来加速训练

best_val = 10000000

# 选择优化方法，这里设置的是 adma 方法
optim = makeOptimizer(model.parameters(), args)

# While training you can press Ctrl + C to stop it.
try:
    print('Training start')
    for epoch in range(1, args.epochs + 1):  #训练次数50次
        epoch_start_time = time.time()

        #print("X=", Data.train[0].shape)
        #print("Y=", Data.train[1].shape)

        train_loss = train_method(Data, Data.train[0], Data.train[1], model, criterion, optim, args)
        #print("____________1__")

        val_rmse, val_loss, val_rae, val_corr = evaluate_method(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args)
        print('| end of epoch {:3d} | time used: {:5.2f}s | train_loss {:5.4f} | valid rmse {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.
                format(epoch, (time.time() - epoch_start_time), train_loss, val_rmse , val_loss, val_rae, val_corr))

        if val_loss < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_loss
        if epoch % 10 == 0:
            test_rmse, test_acc, test_rae, test_corr = evaluate_method(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,args)
            print("| test rmse {:5.4f} | test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}\n".format(test_rmse, test_acc, test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
test_rmse, test_acc, test_rae, test_corr = evaluate_method(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,args)
print('Best model performance：')
print("| test rmse {:5.4f} | test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_rmse, test_acc, test_rae, test_corr))

plot_method(Data, Data.test[0], Data.test[1], model, args)