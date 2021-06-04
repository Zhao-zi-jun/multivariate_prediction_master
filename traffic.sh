#!/usr/bin/env bash

# 使用LSTNet-Annt 方法，horizon=0  滑动窗口7  （去改skip=2）
python main.py --model MHA_Net --horizon 0 --data data/traffic.txt --save save/traffic1.pt --window 7 --CNN_kernel 3 --highway_window 7

# 使用LSTNet-Annt 方法，horizon=0  滑动窗口288（1天）  （去改skip=24）
python main.py --model MHA_Net --horizon 0 --data data/traffic.txt --save save/traffic2.pt --window 288

# 使用LSTNet-Annt 方法，horizon=0  滑动窗口2016=288*7 （去改skip=24）
python main.py --model MHA_Net --horizon 0 --data data/traffic.txt --save save/traffic3.pt --window 2016


# 使用LSTNet-Annt 方法，horizon=12  滑动窗口7   （去改skip=2）
#python main.py --model MHA_Net --horizon 12 --data data/traffic.txt --save save/traffic4.pt --window 7 --CNN_kernel 3 --highway_window 7

# 使用LSTNet-Annt 方法，horizon=12  滑动窗口288 （去改skip=24）
#python main.py --model MHA_Net --horizon 12 --data data/traffic.txt --save save/traffic5.pt --window 288

# 使用LSTNet-Annt 方法，horizon=12  滑动窗口2016 （去改skip=24）
#python main.py --model MHA_Net --horizon 12 --data data/traffic.txt --save save/traffic6.pt --window 2016


# 使用LSTNet-skip 方法，horizon=0  滑动窗口7   （去改skip=2）
#python main.py --horizon 0 --data data/traffic.txt --save save/traffic7.pt --window 7 --CNN_kernel 3 --highway_window 7

# 使用LSTNet-skip 方法，horizon=0  滑动窗口288 （去改skip=24）
#python main.py --horizon 0 --data data/traffic.txt --save save/traffic8.pt --window 288

# 使用LSTNet-skip 方法，horizon=0  滑动窗口2016 （去改skip=24）
#python main.py --horizon 0 --data data/traffic.txt --save save/traffic9.pt --window 2016


# 使用LSTNet-skip 方法，horizon=12  滑动窗口7  （去改skip=2）
python main.py --horizon 12 --data data/traffic.txt --save save/traffic10.pt --window 7 --CNN_kernel 3 --highway_window 7

# 使用LSTNet-skip 方法，horizon=12  滑动窗口288  （去改skip=24）
python main.py --horizon 12 --data data/traffic.txt --save save/traffic11.pt --window 288

# 使用LSTNet-skip 方法，horizon=12  滑动窗口2016  （去改skip=24）
python main.py --horizon 12 --data data/traffic.txt --save save/traffic12.pt --window 2016