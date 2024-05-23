#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:04:36 2024

@author: fandaojin
"""
# import 
import math
import matplotlib.pyplot as plt
from SuperConductingSimulationEnergy import *

#%% XYnoise & XYlength 通过互感计算
att_line = [2.21+11.2*0.6+4, 3.45, 3.46, 2.61, 0, 0] # 线上衰减
att_list = [0, 0, 13, 0, 0, 26] #衰减器+滤波器
att_list = [a1+a2 for a1,a2 in zip(att_line,att_list)]
Mx = 0.3e-12 # XY 互感
g = cal_g_by_Mx(Mx) #通过互感计算g
# print(f'gv: {g:.5e}')
p_th = cal_p_thermal_by_g(
        base_temp = 1.2e3, #初始信号噪声 
        g=g, 
        f=5e9, # 比特频率
        gamma_T1=1/70e-6, # T1
        att_list = att_list,#各层衰减器
        t0_list = [300,50, 4,0.9,0.1,0.015] #各层温度
        )

print(f'p_th:{p_th}')
xy_length = cal_XY_length_by_g(
        xy_dBm = -7,
        xy_amp_percent = 0.6, #X/2门幅度
        f=5e9, # 比特频率
        att_list = att_list,#各层衰减器
        g_v=g,#比特XY线互感
        )
print(f'X/2 length:{xy_length} ns')

#%% Z noise & max_phi0通过互感计算 qubit
att_line = [1+6+1.6+10,1.73,1.72,1.3,0,0] #室温加10dB可调衰减器
att_list = [0,0,13,0,0,0]
att_list = [a1+a2 for a1,a2 in zip(att_line,att_list)]
att_all = sum(att_list)
M_q  = 0.9e-12
Tphi1 = cal_Tphi1(
        base_temp = 4.5e4,
        f01_max=5.3e9,#比特最大频率
        f01=5e9, #比特idle频率
        # M_q=1.07e-12,#比特Z线互感,
        M_q = M_q,
        att_list = att_list,
        t0_list = [300,50,4,0.9,0.1,0.015],
        )
print(f'Tphi1:{Tphi1*1e3} ms')

phi,f01_max_detune = cal_phi_by_Mq(
        f01_max=5.3e9,
        M_q=M_q,
        att_list = att_list,
        Z_amplitude = 0.7, #电子学最大输出
        )
print(f'num_phi0:{phi}',f'f01_max_detune:{f01_max_detune/1e9}')
#%% Z noise & max_phi0通过互感计算 coupler
att_line = [1+6+1.6+4,1.73,1.72,1.3,0,0] #室温加4dB可调衰减器
att_list = [0,0,13,0,0,0]
att_list = [a1+a2 for a1,a2 in zip(att_line,att_list)]
att_all = sum(att_list)
M_q  = 0.9e-12
f01_max = 9.834e9
f01=6.796e9, #coupler=1.22us->qubit=1ms, 比特idle频率,8.75e9, 
f01 = 8.75e9
Tphi1 = cal_Tphi1(
        base_temp = 4.5e4,
        f01_max=f01_max,#比特最大频率
        f01=f01, #coupler=1.22us->qubit=1ms, 比特idle频率,8.75e9, 
        # M_q=1.07e-12,#比特Z线互感,
        M_q = M_q,
        att_list = att_list,
        t0_list = [300,50,4,0.9,0.1,0.015],
        )
print(f'Tphi1:{Tphi1*1e3} ms')
print(f'-----------------------')
phi,f01_max_detune = cal_phi_by_Mq(
        f01_max=f01_max,
        M_q=M_q,
        att_list = att_list,
        Z_amplitude = 0.7, #电子学最大输出
        )
print(f'num_phi0:{phi}',f'f01_max_detune:{f01_max_detune/1e9}')
#%%
Tphi1 = coupler2qubit
#%% 上海Zu3 XYnoise 根据XY互感计算
att_line = [0,1,4,4,4,0] 
att_list = [0,3,10,20,0,16]
att_list = [a1+a2 for a1,a2 in zip(att_line,att_list)]
Mz = e-12
g = cal_g_by_Mx(Mx)
p_th = cal_p_thermal_by_g(
        base_temp = 1.2e4,
        g=g,
        f=5.07e9,
        gamma_T1=1/70e-6,
        att_list = att_list,
        t0_list = [300,50, 4,0.9,0.1,0.015])
print(f'p_th:{p_th}')
#%% 上海Zu3 XYnoise 通过幅度计算
xy_dbm = -7 #需要实测
p_th = cal_p_thermal(
    base_temp = 1.2e4, #初始信号噪声
    xy_dbm =xy_dbm, #满幅输出功率
    xy_amp_percent = 0.45, #X/2门幅度,需要实测
    xy_len_mean = 30e-9, #X/2门长度，需要实测
    f=5.014e9, # 比特频率
    gamma_T1=1/70e-6,  # T1
    att_list = att_list,#各层衰减器
    t0_list = [300,50, 4,0.9,0.1,0.015] #各层温度
    )
print(f'p_th:{p_th}')

#%%
Vpp=0.12
XY_dBm = 20*np.log10(Vpp**2/50/2*1e3)
print(XY_dBm)