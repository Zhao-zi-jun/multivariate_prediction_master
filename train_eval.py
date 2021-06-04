import math
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

def evaluate_method(data, X, Y, model, evaluateL2, evaluateL1, args):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    #print('Y.shape=',Y.shape)
    #print('X.shape=',X.shape)

    ###############
    #y_true =
    #y_predict =
    ##############
    for X, Y in data.get_batches(X, Y, args.batch_size, False):
        output = model(X)
        if predict is None:
            predict = output.clone().detach()
            test = Y
        else:
            predict = torch.cat((predict, output.clone().detach()))
            test = torch.cat((test, Y))
        scale = data.scale.expand(output.size(0), data.m)
        total_loss += float(evaluateL2(output * scale, Y * scale).data.item())
        total_loss_l1 +=float( evaluateL1(output * scale, Y * scale).data.item())

        n_samples += int((output.size(0) * data.m))

    rmse = math.sqrt(total_loss / n_samples)
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    #print('Ytest==',Ytest.shape)
    #print('predict==', predict.shape)
    #print(Ytest[0:1].shape)
    return rmse, rse, rae, correlation

def plot_method(data, X, Y, model, args):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, args.batch_size, False):
        output = model(X)
        if predict is None:
            predict = output.clone().detach()
            test = Y
        else:
            predict = torch.cat((predict, output.clone().detach()))
            test = torch.cat((test, Y))


    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    plt.plot(np.arange(Ytest.shape[1]), Ytest[0:1].reshape(Ytest.shape[1], 1), 'b-', label='true value')
    plt.plot(np.arange(Ytest.shape[1]), predict[0:1].reshape(Ytest.shape[1], 1), 'g-', label='predict value')
    #plt.plot(np.arange(Ytest.shape[1]), Ytest[0:1].reshape(1,Ytest.shape[1]), 'b-', label='true value')
    #plt.plot(np.arange(Ytest.shape[1]), predict[0:1].reshape(1,Ytest.shape[1],), 'g-', label='predict value')
    plt.legend()
    plt.show()



def train_method(data, X, Y, model, criterion, optim, args):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, args.batch_size, True):
        optim.zero_grad()
        #print("X=", X.shape)
        #print("Y=", Y.shape)
        output = model(X)
        #print("______111_______")
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)   # criterion 计算MSE
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optim.step()
        total_loss += loss.data.item()
        n_samples +=int( (output.size(0) * data.m))
    return total_loss / n_samples

# 选择pytorch不同的优化器，这里使用设置好的 adma
def makeOptimizer(params, args):
    if args.optim == 'sgd':   # spg 随机梯度下降
        optimizer = optim.SGD(params, lr=args.lr, )
    #AdaGrad和Adam都属于Per-parameter adaptive learning rate methods（逐参数适应学习率方法）
    #spg 方法是对所有的参数都是一个学习率，AdaGrad和Adam 对不同的参数有不同的学习率。
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr, )
    elif args.optim == 'adadelta':       # Adadelta是Adagrad的改进。
        optimizer = optim.Adadelta(params, lr=args.lr, )
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, )
    else:
        raise RuntimeError("Invalid optim method: " + args.method)
    return optimizer



def evaluate_method(data, X, Y, model, evaluateL2, evaluateL1, args):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    for X, Y in data.get_batches(X, Y, args.batch_size, False):
        output = model(X)
        if predict is None:
            predict = output.clone().detach()
            test = Y
        else:
            predict = torch.cat((predict, output.clone().detach()))
            test = torch.cat((test, Y))
        scale = data.scale.expand(output.size(0), data.m)
        total_loss += float(evaluateL2(output * scale, Y * scale).data.item())
        total_loss_l1 +=float( evaluateL1(output * scale, Y * scale).data.item())
        n_samples += int((output.size(0) * data.m))

    rmse = math.sqrt(total_loss / n_samples)
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rmse, rse, rae, correlation