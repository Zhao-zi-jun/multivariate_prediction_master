#!/usr/bin/env bash
# 使用LSTNet-Annt 方法，horizon=0  滑动窗口7 （去改skip=2）
python main.py --model MHA_Net --horizon 0 --data data/commodity.txt --save save/commod1.pt --window 7 --CNN_kernel 3 --highway_window 7

# 使用LSTNet-Annt 方法，horizon=0  滑动窗口10 （去改skip=2）
python main.py --model MHA_Net --horizon 0 --data data/commodity.txt --save save/commod2.pt --window 10 --CNN_kernel 3 --highway_window 10

# 使用LSTNet-Annt 方法，horizon=0  滑动窗口24 （去改skip=12）
python main.py --model MHA_Net --horizon 0 --data data/commodity.txt --save save/commod3.pt --window 24

# 使用LSTNet-Annt 方法，horizon=12  滑动窗口7 （去改skip=2）
python main.py --model MHA_Net --horizon 12 --data data/commodity.txt --save save/commod4.pt --window 7 --CNN_kernel 3 --highway_window 7

# 使用LSTNet-Annt 方法，horizon=12  滑动窗口10 （去改skip=2）
python main.py --model MHA_Net --horizon 12 --data data/commodity.txt --save save/commod5.pt --window 10 --CNN_kernel 3 --highway_window 10

# 使用LSTNet-Annt 方法，horizon=12  滑动窗口24 （去改skip=12）
python main.py --model MHA_Net --horizon 12 --data data/commodity.txt --save save/commod6.pt --window 24



# 使用LSTNet-skip 方法，horizon=0  滑动窗口7 （去改skip=2）
python main.py --model LSTNet --horizon 0 --data data/commodity.txt --save save/commod7.pt --window 7 --CNN_kernel 3 --highway_window 7

# 使用LSTNet-skip 方法，horizon=0  滑动窗口10 （去改skip=2）
python main.py --model LSTNet --horizon 0 --data data/commodity.txt --save save/commod8.pt --window 10 --CNN_kernel 3 --highway_window 10

# 使用LSTNet-skip 方法，horizon=0  滑动窗口24 （去改skip=12）
#python main.py --model LSTNet --horizon 0 --data data/commodity.txt --save save/commod9.pt --window 24



# 使用LSTNet-skip 方法，horizon=12  滑动窗口7 （去改skip=2）
#python main.py --model LSTNet --horizon 0 --data data/commodity.txt --save save/commod10.pt --window 7 --CNN_kernel 3 --highway_window 7

# 使用LSTNet-skip 方法，horizon=12  滑动窗口10 （去改skip=2）
#python main.py --model LSTNet --horizon 0 --data data/commodity.txt --save save/commod11.pt --window 10 --CNN_kernel 3 --highway_window 10

# 使用LSTNet-skip 方法，horizon=12  滑动窗口24 （去改skip=12）
#python main.py --model LSTNet --horizon 0 --data data/commodity.txt --save save/commod12.pta --window 24