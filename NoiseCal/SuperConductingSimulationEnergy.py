#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:31:55 2024

@author: fandaojin
"""

import numpy as np  
from scipy.linalg import inv  
from scipy.linalg import kron  
from scipy.special import factorial  
import cmath  
from scipy.linalg import eig  
import math  

import sys
import os
import matplotlib.pyplot as plt


def R2L(R):  
    # R的单位是欧姆，Ej的单位是Hz  
    e = 1.60217e-19  # 电子电荷量，单位库仑  
    h = 6.6260e-34  # 普朗克常数，单位焦耳秒  
    phi0 = h / (2 * e)  # 磁通量子，单位韦伯  
    Ij = 2.8e-4 / R  # 电流，单位安培  
    Lj = phi0 / (2 * math.pi * Ij)  # 电感，单位亨利  
    # 注意：根据原函数名和上下文，这里计算的可能不是电感(Lj)，而是与能量量子Ej有关的量  
    # 但是，由于原函数名为R2Ej，这里假设我们需要计算与Ej相关的值，并将其命名为Lj（可能是一个错误）  
    # 如果实际上要计算电感，则函数名应该更改为反映其实际功能的名称  
    return Lj  
  
  
def Rphi2Lj(R1, R2, phi):  
    alpha = min(R1, R2) / max(R1, R2)  
    Lj1 = R2L(min(R1, R2)) / 2  
    # real_part = Lj1 / (abs(1 + alpha) / 2) * cmath.cos(cmath.pi * phi)  
    # imag_part = 1j * Lj1 / (abs(1 - alpha) / 2) * cmath.sin(cmath.pi * phi)  
    # Lj = real_part + imag_part  
    Lj = Lj1/abs((1+alpha)/2*np.cos(np.pi*phi)+1j*(1-alpha)/2*np.sin(np.pi*phi))
    return Lj    
  
def H_qubit(Ej, Ec, n, Driven=None):  
    if Driven is None:  
        Driven = np.array([0.0, 0.0], dtype=float)  # 确保Driven是浮点数类型的NumPy数组  
      
    DrivenX, DrivenY = Driven  
    cos_num = [2, 4, 6, 8]  
    n_add = n + 5  
    H_add = np.zeros((n_add, n_add), dtype=complex)  # 明确指定H_add为复数类型  
    a = np.zeros((n_add, n_add), dtype=complex)  # 明确指定a为复数类型  
      
    for ii in range(1, n_add):  
        a[ii-1, ii] = np.sqrt(ii)  
      
    X = (a + a.T) / np.sqrt(2)  
    Y = 1j * (a - a.T) / np.sqrt(2)  
      
    E_basis = np.sqrt(8 * Ej * Ec)  
    # 使用np.newaxis来广播DrivenX和DrivenY到与np.eye(n_add)相同的形状  
    H_add += 0.5 * E_basis * np.dot((Y - np.dot(DrivenY , np.eye(n_add))), (Y - np.dot(DrivenY , np.eye(n_add))))- 0.5 * E_basis * np.eye(n_add)  
      
    for jj in range(len(cos_num)):  
        # 直接使用DrivenX标量值，它会被广播到与X相同的形状  
        driven_term = DrivenX * np.eye(n_add)  
        # 从X中减去广播后的DrivenX  
        x_minus_driven = X - driven_term  
        # 进行幂运算，此时x_minus_driven和cos_num[jj]都是数组  
        term = x_minus_driven
        for kk in range(cos_num[jj]-1):
            term = np.dot(x_minus_driven,term)
        H_add -= (-1)**(jj-1) * Ej / factorial(cos_num[jj]) * (8 * Ec / Ej)**(cos_num[jj] / 4) * term
    H = H_add[:n, :n]  
    return H
  
# 在 HSC 函数中使用 H_qubit  
# ...  
# 假设H_cavity, H_c_interact, H_j_interact以及tensor_product都有相应的Python实现  
def H_cavity(Ej, Ec, n, is_nonlinear=False):  
    H = np.zeros((n, n))  
    H[0, 0] = 0  
    for ii in range(1, n):  
        H[ii, ii] = H[ii-1, ii-1] + np.sqrt(8 * Ej * Ec) - (ii - 1) * Ec * is_nonlinear  
    return H  
  
# 使用 H_cavity 函数  
# Ej, Ec, n 是你提供的参数  
# H = H_cavity(Ej, Ec, n)
  
def tensor_product(M_in, Dim=None):  
    # 当只提供M_in参数时  
    if Dim is None:  
        if len(M_in) == 0:  
            M_out = np.eye(1)  
        else:  
            M_out = M_in[0]  
            for ii in range(1,len(M_in)):
            # for matrix in M_in[1:]:  
                M_out = kron(M_out, M_in[ii])  
        return M_out  
      
    # 当同时提供M_in和Dim参数时  
    M_entanglement = M_in[0]  
    N1, N2, N12 = Dim  
    # M_entanglement = M_in[1]  
    # N1 = Dim(1)  
    # N2 = Dim(2)  
    # N12 = Dim(3)  
      
    # 初始化M_depart为一个字典，键为(ii, jj)的元组，值为NumPy数组  
    M_depart = {(ii, jj): np.zeros((N2, N2)) for ii in range(N1) for jj in range(N1)}  
      
    # 初始化M_out为一个NumPy数组  
    M_out = np.zeros((N1 * N2 * N12, N1 * N2 * N12))  
      
    # MATLAB中的循环，用Python的for循环来替代  
    for ii in range(N1):  
        for jj in range(N1):  
            # MATLAB中的 M_depart{ii,jj}=M_entanglement((1:N2)+N2*(ii-1),(1:N2)+N2*(jj-1));  
            # start_idx = N2 * ii
            # end_idx = start_idx + N2  
            M_depart[(ii, jj)] = M_entanglement[N2 * ii:N2 * (ii+1), N2*jj:N2*(jj+1)]  
              
            # MATLAB中的 M_depart{ii,jj}=kron(M_in{2},M_depart{ii,jj});  
            M_depart[(ii, jj)] = np.kron(M_in[1], M_depart[(ii, jj)])  
              
            # MATLAB中的 M_out((1:N2*N12)+(ii-1)*N2*N12,(1:N2*N12)+(jj-1)*N2*N12)=M_depart{ii,jj};  
            M_out[ii * N2 * N12: (ii + 1) * N2 * N12,   
                  jj * N2 * N12: (jj + 1) * N2 * N12] = M_depart[(ii, jj)]
      
    return M_out  
  
# 示例使用  
# M_in是一个列表，包含numpy数组  
# Dim是一个包含三个元素的元组，表示维度  
# M_out = tensor(M_in, Dim)

def H_c_interact(Ec_c, Ec_1, Ej_1, Ec_2, Ej_2, n1, n2):  
    # 创建对角线下方的矩阵a1和a2  
    a1 = np.zeros((n1, n1), dtype=complex)  
    for ii in range(2, n1 + 1):  
        a1[ii - 2, ii - 1] = np.sqrt(ii - 1)  
  
    a2 = np.zeros((n2, n2), dtype=complex)  
    for ii in range(2, n2 + 1):  
        a2[ii - 2, ii - 1] = np.sqrt(ii - 1)  
  
    # 计算k1和k2  
    k1 = np.kron((Ej_1 / (8 * Ec_1))**(1/4) * (a1.conj().T - a1) / np.sqrt(2) / 1j, np.eye(n2))  
    k2 = np.kron(np.eye(n1), (Ej_2 / (8 * Ec_2))**(1/4) * (a2.conj().T - a2) / np.sqrt(2) / 1j)  
  
    # 计算最终的H  
    H = 4 * Ec_c * (k1.dot(k2) + k2.dot(k1))  
  
    return H  

def H_j_interact(Ej_c, Ec_1, Ej_1, Ec_2, Ej_2, n1, n2):  
    # 创建对角线下方的矩阵a1和a2  
    a1 = np.zeros((n1, n1), dtype=complex)  
    for ii in range(2, n1 + 1):  
        a1[ii - 2, ii - 1] = np.sqrt(ii - 1)  
  
    a2 = np.zeros((n2, n2), dtype=complex)  
    for ii in range(2, n2 + 1):  
        a2[ii - 2, ii - 1] = np.sqrt(ii - 1)  
  
    # 计算phi1和phi2  
    phi1 = np.kron((8 * Ec_1 / Ej_1)**(1/4) * (a1.conj().T + a1) / np.sqrt(2), np.eye(n2))  
    phi2 = np.kron(np.eye(n1), (8 * Ec_2 / Ej_2)**(1/4) * (a2.conj().T + a2) / np.sqrt(2))  
  
    # 计算最终的H  
    H = 0.5 * Ej_c * (phi1.dot(phi2) + phi2.dot(phi1))    
    return H  
  
def HSC(M_C, M_L, N, n_Points, is_nonlinear=None, DrivenXY=None):  
    if DrivenXY is None:  
        DrivenXY = np.zeros((n_Points, 2))
      
    M_C_new = np.zeros((n_Points, n_Points))  
    M_Ej_0 = 6.62e-34 / (4 * np.pi**2 * 4 * (1.6e-19**2)) / 1e9 / M_L  
    M_Ej = np.zeros((n_Points, n_Points))  
      
    for ii in range(n_Points):  
        for jj in range(n_Points):  
            if ii == jj:  
                M_C_new[ii, jj] = np.sum(M_C[:, ii])  
                M_Ej[ii, jj] = np.sum(M_Ej_0[:, ii])  
            else:  
                M_C_new[ii, jj] = -M_C[ii, jj]  
                M_Ej[ii, jj] = -M_Ej_0[ii, jj]  
      
    M_Ec = 1.6e-19**2 / 2 / 6.62e-34 / 1e9 * np.linalg.inv(M_C_new)  # GHz  
      
    H = {}  
    H0 = {ii: np.eye(N[ii]) for ii in range(n_Points)}  
    dim = 1  
    for ii in range(n_Points):  
        dim *= N[ii]  
      
    H_total = np.zeros((dim, dim), dtype=complex)
      
    for ii in range(n_Points):  
        for jj in range(ii, n_Points):  
            if ii == jj:  
                H_temp = H0.copy()  
                if is_nonlinear is not None and is_nonlinear[ii]:  
                    H_temp[ii] = H_qubit(M_Ej[ii, jj], M_Ec[ii, jj], N[ii], DrivenXY[ii, :])  
                else:  
                    H_temp[ii] = H_cavity(M_Ej[ii, jj], M_Ec[ii, jj], N[ii])  
                H[(ii, jj)] = tensor_product(H_temp)  
            else:  
                H_temp = H_c_interact(M_Ec[ii, jj], M_Ec[ii, ii], M_Ej[ii, ii], M_Ec[jj, jj], M_Ej[jj, jj], N[ii], N[jj]) + H_j_interact(M_Ej[ii, jj], M_Ec[ii, ii], M_Ej[ii, ii], M_Ec[jj, jj], M_Ej[jj, jj], N[ii], N[jj])  
                M_left = tensor_product({kk:H0[kk] for kk in range(ii)})  
                M_right = tensor_product({kk:H0[kk+jj+1] for kk in range(n_Points-(jj + 1))})  
                M_size = 1  
                for nn in range(ii + 1, jj):  
                    M_size *= N[nn]  
                M_middle = tensor_product({0:H_temp, 1:np.eye(M_size)}, [N[ii], N[jj], M_size])  
                H[(ii, jj)] = tensor_product({0:M_left, 1:M_middle, 2:M_right})  
              
            H_total += H[(ii, jj)]   
      
    return H, H_total, M_Ej, M_Ec  



def coth(x):
    return math.cosh(x) / math.sinh(x)
def arccoth(x):
    return np.log(2/(x-1)+1)/2

def blackBodyRadiationTransver(e_t, f, is_energy, Vesion='V2'):
    h=6.623e-34
    c=3e8
    lambda_= c/f
    kb = 1.38e-23
    R=50
    if Vesion == 'V1':
        if is_energy == True:
            return 1/(np.log(1/(e_t/(2*h*c**2/lambda_**5))+1)/h/c*lambda_*kb)
        else:
            return 2*h*c**2/lambda_**5*(1/(np.exp(h*c/lambda_/kb/e_t)-1)) 
    elif Vesion == 'V2':
        if is_energy == True:
            return h*f/(kb*np.log(4*R*h*f/e_t+1))            
        else:
            return 4*R*h*f/(np.exp(h*f/kb/e_t)-1)
    elif Vesion == 'V3':
        if is_energy == True:
            return h*f/(arccoth(e_t/h/f/R-1))/2/kb
        else:
            return h*f*50*((coth(h*f/(2*kb*e_t)))+1)
                      
def calculateEquivalentTemperature(base_temp,att_list,is_reverse=False, t0_list = [300,50,4,1,0.1,0.01],f=1e3):
    # temp_300k_att,temp_50k_att,temp_4k_att,temp_1k_att,temp_cp_att,temp_mxc_att = att
    h=6.623e-34
    c=3e8
    lambda_=c/f
    kb = 1.38e-23
    e_total =  blackBodyRadiationTransver(base_temp, f, False)
    if len(t0_list) != len(att_list):
        return print('error: len(att_list) != len(t0_list)')
    for ii in range(len(t0_list)):
        t0 = t0_list[ii] if is_reverse==False else t0_list[-1-ii]
        if is_reverse==True:
            if base_temp<t0:
                return print(f'error: base_temp in {t0} k = {round(base_temp,2)} k , which is < {t0} K')
        e_total = (e_total/10**(att_list[ii]/10)+(1-1/(10**(att_list[ii]/10)))*blackBodyRadiationTransver(t0_list[ii], f, False)) if is_reverse==False else (e_total-(1-1/(10**(att_list[-1-ii]/10)))*blackBodyRadiationTransver(t0_list[-1-ii], f, False))*10**(att_list[-1-ii]/10)
        base_temp =blackBodyRadiationTransver(e_total/2, -f, True, Vesion='V3')
        
        # e_dbm = 10*np.log10(kb*base_temp*1e3)
        e_dbm = 10*np.log10(e_total/4/50*1e3)
        print(f'Temp after {t0} k = {round(e_dbm,2)} dBm , {round(base_temp,5)} K and S_w_v = {e_total} V**2/Hz') 
        e_total
    return e_total

def calculateEquivalentAttenuator(T_start, T_end, f=5e9):
    h=6.623e-34
    c=3e8
    lambda_=c/f
    kb = 1.38e-23
    e_s = blackBodyRadiationTransver(T_start, f, False)        
    e_end = blackBodyRadiationTransver(T_end, f, False) 
    att = 10*np.log10(e_s/e_end)
    print(f'T_start = {T_start} To T_end = {T_end} is attenuated by {att} dB')
 
def cal_p_thermal(
        base_temp = 1.2e4,
        xy_dbm =-15,
        xy_amp_percent = 0.45,
        xy_len_mean = 30e-9,
        f=5.17e9,
        gamma_T1=1/70e-6,
        att_list = [0,3+1,10+2,20+2,2,16],
        t0_list = [300,50,4,0.9,0.1,0.015],
        ):
    h=6.623e-34
    kb = 1.38e-23
    xy_amp_mean = np.sqrt(10**(xy_dbm/10)*50/1e3*2)*xy_amp_percent
    # print('xy_amp_mean:', xy_amp_mean)
    att_all = np.sum(att_list)
    k_att = 10**(att_all/20)
    g=k_att/(xy_amp_mean*xy_len_mean)*np.pi 
    print(f'gv: {g:.5e}')
    sv = calculateEquivalentTemperature(base_temp,att_list,is_reverse=False, t0_list = t0_list,f=f)
    T = blackBodyRadiationTransver(sv/2, -f, True, Vesion='V3')
    gamma_th_up = g**2*sv/2
    gamma_th_down = gamma_th_up/np.exp(-h*f/kb/T)
    sv_down = sv/2/np.exp(-h*f/kb/T)
    print('sv_down:',sv_down)
    p_th = gamma_th_up/(gamma_T1+gamma_th_down+gamma_th_up)    
    return p_th

def cal_gamma_th_down(
        base_temp = 1.2e4,
        xy_dbm =-15,
        xy_amp_percent = 0.45,
        xy_len_mean = 30e-9,
        f=5.17e9,
        gamma_T1=1/70e-6,
        att_list = [0,3+1,10+2,20+2,2,16],
        t0_list = [300,50,4,0.9,0.1,0.015],
        ):
    h=6.623e-34
    kb = 1.38e-23
    xy_amp_mean = np.sqrt(10**(xy_dbm/10)*50/1e3*2)*xy_amp_percent
    # print('xy_amp_mean:', xy_amp_mean)
    att_all = np.sum(att_list)
    k_att = 10**(att_all/20)
    g=k_att/(xy_amp_mean*xy_len_mean)*np.pi 
    print(f'{g:.5e}')
    sv = calculateEquivalentTemperature(base_temp,att_list,is_reverse=False, t0_list = t0_list,f=f)

    T = blackBodyRadiationTransver(sv/2, -f, True, Vesion='V3')
    gamma_th_up = g**2*sv/2
    gamma_th_down = gamma_th_up/np.exp(-h*f/kb/T)
    sv_down = sv/2/np.exp(-h*f/kb/T)
    print('sv_down:',sv_down)
    p_th = gamma_th_up/(gamma_T1+gamma_th_down+gamma_th_up)    
    return gamma_th_down


def cal_p_thermal_by_g(
        base_temp = 1.2e4,
        g=2.19e12,
        f=5.17e9,
        gamma_T1=1/70e-6,
        att_list = [0,3+1,10+2,20+2,2,16],
        t0_list = [300,50,4,0.9,0.1,0.015],
        ):
    h=6.623e-34
    kb = 1.38e-23
    sv = calculateEquivalentTemperature(base_temp,att_list,is_reverse=False, t0_list = t0_list,f=f)
    T = blackBodyRadiationTransver(sv/2, -f, True, Vesion='V3')
    # T = 30e-3
    gamma_th_up = g**2*sv/2
    gamma_th_down = gamma_th_up/np.exp(-h*f/kb/T)
    p_th = gamma_th_up/(gamma_T1+gamma_th_down+gamma_th_up)    
    return p_th    
 
def calculateM2A(M):
    m_0=0.8
    return 2-20*np.log10(M/m_0)

#% calculateT1 Influenced by excitation
def calculateT1IBE(T1, T_est, T_idel=0.04, freq=5e9):
    kb = 1.38e-23
    h = 6.623e-34
    gamma1=1/T1
    gamma_down =gamma1/(1+np.exp(-h*freq/kb/T_est)) 
    gamma1_idel =gamma_down*(1+np.exp(-h*freq/kb/T_idel)) 
    res0 = np.exp(-h*freq/kb/T_est)
    print(f'excitating rate = {res0}')
    print(f'1/gamma1 = {1/gamma1}')
    print(f'1/gamma_down = {1/gamma_down}')
    print(f'1/gamma1_idel = {1/gamma1_idel}')
def cal_Tphi1_to_Sv2(
        f01_max=4.435e9,#比特最大频率
        f01=4.621e9, #比特idle频率
        Tphi1=1e-3,
        M_q=1.07e-12,#比特Z线互感
        ):
    f01_max=f01_max #比特最大频率
    f01=f01 #比特idle频率
    M_q=M_q#比特Z线互感
    kb = 1.38e-23 
    Phi_0 = 2.067e-15 #单位Wb，h/2e
    phi=np.arccos(((f01)/f01_max)**2)/np.pi # 计算idle频率的量子磁通
    k_f2phi = 2*np.pi*f01_max*np.pi*np.sin(np.pi*phi)/2/np.sqrt(np.cos(phi*np.pi)) #计算idle频率的斜率
    s_w_phi = 4/k_f2phi**2/Tphi1 #Tphi对应的量子磁通功率谱密度函数
    Sv2 = s_w_phi*Phi_0**2/M_q**2*50**2/4 #Tphi1对应的电压功率谱密度函数
    print('Sv2:', Sv2, 's_w_phi:', s_w_phi)
    T_est = blackBodyRadiationTransver(Sv2, f=1e6, is_energy=True, Vesion='V2')
    print('T_est:',T_est)
    print('PSD:', 10*np.log10(kb*T_est*1e3))
    return s_w_phi, Sv2
def cal_Tphi1_by_Sv2(
        f01_max=4.435e9,#比特最大频率
        f01=4.621e9, #比特idle频率
        Sv2=1e-3,
        M_q=1.07e-12,#比特Z线互感
        ):
    M_q=M_q#比特Z线互感
    kb = 1.38e-23 
    Phi_0 = 2.067e-15 #单位Wb，h/2e
    phi=np.arccos(((f01)/f01_max)**2)/np.pi # 计算idle频率的量子磁通
    Z_amplitude =  phi*Phi_0/M_q/2
    k_f2phi = 2*np.pi*f01_max*np.pi*np.sin(np.pi*phi)/2/np.sqrt(np.cos(phi*np.pi)) #计算idle频率的斜率
    s_w_phi = Sv2/(Phi_0**2/M_q**2*50**2/4) #Tphi1对应的电压功率谱密度函数
    Tphi1 = 4/k_f2phi**2/s_w_phi #Tphi对应的量子磁通功率谱密度函数
    print(f'Current:{Z_amplitude*1e3} mA')
    return Tphi1

def cal_Tphi1(
        base_temp = 1.2e4,
        f01_max=4.435e9,#比特最大频率
        f01=4.621e9, #比特idle频率
        M_q=1.07e-12,#比特Z线互感
        att_list = [0,3+1,10+2,20+2,2,16],
        t0_list = [300,50,4,0.9,0.1,0.015],
        ):
    h=6.623e-34
    kb = 1.38e-23
    sv = calculateEquivalentTemperature(base_temp,att_list,is_reverse=False, t0_list = t0_list,f=f01)
    Tphi1 = cal_Tphi1_by_Sv2(
        f01_max=f01_max,#比特最大频率
        f01=f01, #比特idle频率
        Sv2=sv,
        M_q=M_q,#比特Z线互感
        )   
    return Tphi1
    
def get_k(zpa2f01, control, fah=-230e6):
    maxF01,biasOffset, kf = zpa2f01
    pi=np.pi
    k = -pi*kf*(maxF01-fah)/2/np.sqrt(abs(np.cos(pi*kf*(control-biasOffset))))*np.sin(pi*kf*(control-biasOffset))
    return k

def get_control(zpa2f01, f01 = 3.511e9, fah =-230e6):
    maxF01,biasOffset, kf = zpa2f01
    control = np.arccos(((f01-fah)/(maxF01-fah))**2)/np.pi/kf+biasOffset
    return control

def get_f01_by_zpa2f01(control_list, zpa2f01, fah =-230e6):
    maxF01,biasOffset, kf = zpa2f01
    pi=np.pi
    return [(maxF01-fah)*np.sqrt(abs(np.cos(pi*kf*(control-biasOffset))))+fah for control in control_list]

def cal_g_by_Mx(Mx=0.94e-12):  
    n_Points = 3  
    N = np.array([5, 5, 5])  
    isQubit = np.array([1, 1, 1])  
    e = 1.6e-19  # 电子电荷量  
    h = 6.62e-34  # 普朗克常数  
    phi0 = h / (2 * e)  # 磁通量子  
      
    # 定义约瑟夫森结参数  
    Ra = 8800  
    Rb = 8800  
    phiqa = 0  
    phiqb = 0  
    Rc1 = 3000  
    Rc2 = 3000  
    Rc = 1 / (1 / Rc1 + 1 / Rc2)  
    # 定义电容参数  
    Cqc = 10.11e-15  
    Cqq = 0.5e-15  
    Cq = 78.1e-15 + 10e-15 - 10e-15
    Cc = 50.4e-15 + 10e-15 * Ra / Rc  
    # 电容电感矩阵初始化  
    M_C = np.array([[Cq, Cqq, Cqc],  
                     [Cqq, Cq, Cqc],  
                     [Cqc, Cqc, Cc]])  
    M_L = np.ones((3, 3))  
    M_C += 1e-22  # 添加一个小量以避免奇异矩阵  
    phiOffset = np.array([phiqa, phiqb, 0])  
    M_L[0, 0] = Rphi2Lj(Ra * 2, Ra * 2, phiOffset[0])  
    M_L[1, 1] = Rphi2Lj(Rb * 2, Rb * 2, phiOffset[1])  
    M_L[2, 2] = Rphi2Lj(Rc1, Rc2, phiOffset[2])  
    DrivenXY = np.zeros((n_Points, 2))  
    # 调用HSC函数  
    H, H_total, M_Ej, M_Ec = HSC(M_C, M_L, N, n_Points, isQubit, DrivenXY)  
      
    # 计算静态状态和能量  
    energy_static, state_static =np.linalg.eig(H_total)  
    # 对本征值进行排序，并获取排序后的索引  
    sorted_indices = np.argsort(energy_static)  
    # 使用排序后的索引重新排列本征值  
    sorted_energy_static = energy_static[sorted_indices]  
    # 使用相同的索引重新排列本征向量（每一列是一个本征向量）  
    sorted_state_static = state_static[:, sorted_indices]  
    energy = sorted_energy_static - sorted_energy_static[0]
    Us = sorted_state_static  
      
    # 提取特定频率  
    f01qa = energy[1]  
    f01qb = energy[2]  
    
    # 提取Ej和Ec  
    Ej = M_Ej[0][0]  
    Ec = M_Ec[0][0]  
    print('f01:', f01qa, 'anharmonic:', Ec)
    # 根据互感计算g
    gI=2*np.pi*Mx*Ej*(2*Ec/Ej)**(1/4)/phi0*1e9*2*np.pi*2
    print()
    gv=gI/50
    return gv
def cal_XY_length_by_g(
        xy_dBm = -7,
        xy_amp_percent = 0.45, #X/2门幅度
        f=5.014e9, # 比特频率
        att_list = [0,3+1,10+2,20+2,2,16],#各层衰减器
        g_v=2e12,#比特XY线互感对应的g
        ):
    h=6.623e-34
    kb = 1.38e-23
    xy_amp_mean = np.sqrt(10**(xy_dBm/10)*50/1e3*2)*xy_amp_percent
    att_all = np.sum(att_list)
    k_att = 10**(att_all/20)
    xy_len_mean=k_att/(xy_amp_mean*g_v)*np.pi 
    return xy_len_mean*1e9 #ns
def cal_phi_by_Mq(
        f01_max=5.3e9,
        M_q=1.07e-12,
        att_list = [0,3+1,10+2,20+2,2,16],
        Z_amplitude = 1, #电子学最大输出
        ):
    # att_all = np.sum(att_list)
    kb = 1.38e-23 
    Phi_0 = 2.067e-15 #单位Wb，h/2e
    att_all = np.sum(att_list)
    k_att = 10**(att_all/20)
    Z_mc = Z_amplitude/k_att
    phi = Z_mc/(Phi_0/M_q*50/2)
    f01_max_detune = f01_max - f01_max*np.sqrt(np.abs(np.cos(phi*np.pi)))
    print(f"max_Current:{Z_mc/50*1e3}mA")
    return phi,f01_max_detune