from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt

def z_pacf(data):
    x = data[:, 1]
    acf_x, interval = pacf(x=x, nlags=30, alpha=0.05)
    print('ACF:\n', acf_x)
    print('ACF95%置信区间下界:\n', interval[:, 0] - acf_x)
    print('ACF95%置信区间上界:\n', interval[:, 1] - acf_x)
    plot_pacf(x=x, lags=100, alpha=0.05)
    plt.show()

def z_acf(data):
    x = data[:, 1]
    acf_x, interval = acf(x=x, nlags=30, alpha=0.05)
    print('ACF:\n', acf_x)
    print('ACF95%置信区间下界:\n', interval[:, 0] - acf_x)
    print('ACF95%置信区间上界:\n', interval[:, 1] - acf_x)
    plot_acf(x=x, lags=1000, alpha=0.05)
    plt.show()


import pandas as pd
import matplotlib as mpl
import numpy as np


data = pd.read_csv("data/traffic.txt")
data = np.array(data)

print(data)
z_pacf(data)


