#!/usr/bin/env bash

# 使用LSTNet-Annt 方法，horizon=0  (默认highway_window=24) 滑动窗口168=24*7
#python main.py --model MHA_Net --horizon 0 --data data/electricity.txt --save save/elec1.pt --window 168

# 使用LSTNet-Annt 方法，horizon=0   滑动窗口24 （去改skip=12）
#python main.py --model MHA_Net --horizon 0 --data data/electricity.txt --save save/elec2.pt --window 24 --CNN_kernel 3

# 使用LSTNet-Annt 方法，horizon=12  滑动窗口168=24*7
#python main.py --model MHA_Net --horizon 12 --data data/electricity.txt --save save/elec3.pt --window 168

# 使用LSTNet-Annt 方法，horizon=12  滑动窗口24 （去改skip=12）
#python main.py --model MHA_Net --horizon 12 --data data/electricity.txt --save save/elec4.pt --window 24 --CNN_kernel 3


# 使用LSTNet-skip 方法，horizon=0  (默认skip=24) 滑动窗口168=24*7
python main.py --model LSTNet --horizon 0 --data data/electricity.txt --save save/elec5.pt --window 168

# 使用LSTNet-skip 方法，horizon=0  (默认skip=24) 滑动窗口24  （去改skip=12）
python main.py --model LSTNet --horizon 0 --data data/electricity.txt --save save/elec6.pt --window 24 --CNN_kernel 3

# 使用LSTNet-skip 方法，horizon=12  (默认skip=24) 滑动窗口168=24*7
python main.py --model LSTNet --horizon 12 --data data/electricity.txt --save save/elec7.pt --window 168

# 使用LSTNet-skip 方法，horizon=12  (默认skip=24) 滑动窗口24  （去改skip=12）
#python main.py --model LSTNet --horizon 12 --data data/electricity.txt --save save/elec8.pt --window 24 --CNN_kernel 3
