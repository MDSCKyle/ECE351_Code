################################################################
# Kyle McCain                                                  #
# ECE351-53                                                    #
# LAB 9                                                        #
# Due 3/30/2021                                                #
# Fourier Series Square Wave Approximation                     #
################################################################

import numpy as np
import scipy.signal as sig
import scipy.fftpack as ft
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



def usr_FFT(x,fs):
    N = len(x)
    X_fft = ft.fft(x)
    X_fft_shifted = ft.fftshift(X_fft)
    freq = np.arange(-N/2,N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return (X_mag,X_phi,freq)

#Task 1 - fft for cos(2pi*t)
steps = 1.0/100.0
t = np.arange(0, 2, steps)
x = np.cos(2*np.pi*t)
fig = plt.figure(constrained_layout = True, figsize=(10,8))
gs = GridSpec(3,2,figure = fig)
p1 = fig.add_subplot(gs[0,:])
plt.plot(t,x)
plt.grid()
plt.ylabel("x(t)")
plt.xlabel("t")
plt.title("Task 1 - User Defined FFT of x(t)")
xfft = usr_FFT(x,100)
p2 = fig.add_subplot(gs[1,0])
plt.stem(xfft[2],xfft[0])
plt.grid()
plt.ylabel("|X(f)|")
p3 = fig.add_subplot(gs[1,1])
plt.stem(xfft[2],xfft[0])
plt.axis([-2.1, 2.1, -0.1, 0.6])
plt.grid()
p4 = fig.add_subplot(gs[2,0])
plt.stem(xfft[2],xfft[1])
plt.grid()
plt.ylabel("angle X(f)")
plt.xlabel("f [Hz]")
p5 = fig.add_subplot(gs[2,1])
plt.stem(xfft[2],xfft[1])
plt.axis([-2.1, 2.1, -4, 4])
plt.grid()

def fnz(a): #first approximate nonzero in array
    idx = 0
    for i in a:
        if i > 0.001:
            return idx
        idx += 1
    return 0

def lnz(a):
    idx = 0
    last = 0
    for i in a:
        if i > 0.001:
            last = idx
        idx += 1
    return last

def plt_usr_FFT(x,fs,t1,t2,title):
    fig = plt.figure
    steps = 1.0/fs
    t = np.arange(t1, t2, steps)
    fig = plt.figure(constrained_layout = True, figsize=(10,8))
    gs = GridSpec(3,2,figure = fig)
    fig.add_subplot(gs[0,:])
    plt.plot(t,x)
    plt.grid()
    plt.ylabel("x(t)")
    plt.xlabel("t")
    plt.title(title)
    xfft = usr_FFT(x,100)
    fig.add_subplot(gs[1,0])
    plt.stem(xfft[2],xfft[0])
    plt.grid()
    plt.ylabel("|X(f)|")
    fig.add_subplot(gs[1,1])
    plt.stem(xfft[2],xfft[0])
    #print(fnz(xfft[0]),xfft[0][fnz(xfft[0])],xfft[2][fnz(xfft[0])])
    plt.axis([ xfft[2][fnz(xfft[0])]-1,xfft[2][lnz(xfft[0])]+1, min(xfft[0])-0.5, max(xfft[0])+0.5])
    plt.grid()
    fig.add_subplot(gs[2,0])
    plt.stem(xfft[2],xfft[1])
    plt.grid()
    plt.ylabel("angle X(f)")
    plt.xlabel("f [Hz]")
    fig.add_subplot(gs[2,1])
    plt.stem(xfft[2],xfft[1])
    plt.axis([ xfft[2][fnz(xfft[0])]-1,xfft[2][lnz(xfft[0])]+1, min(xfft[1])-0.5, max(xfft[1])+0.5])
    plt.grid()
    plt.xlabel("f [Hz]")

def plt_usr_FFT2(x,fs,t1,t2,title):
    fig = plt.figure
    steps = 1.0/fs
    t = np.arange(t1, t2, steps)
    fig = plt.figure(constrained_layout = True, figsize=(10,8))
    gs = GridSpec(3,2,figure = fig)
    fig.add_subplot(gs[0,:])
    plt.plot(t,x)
    plt.grid()
    plt.ylabel("x(t)")
    plt.xlabel("t")
    plt.title(title)
    xfft = usr_FFT(x,100)
    for i in range(len(xfft[0])):
        if xfft[0][i] < 1e-10:
            xfft[0][i] = 0
            xfft[1][i] = 0
    fig.add_subplot(gs[1,0])
    plt.stem(xfft[2],xfft[0])
    plt.grid()
    plt.ylabel("|X(f)|")
    fig.add_subplot(gs[1,1])
    plt.stem(xfft[2],xfft[0])
    #print(fnz(xfft[0]),xfft[0][fnz(xfft[0])],xfft[2][fnz(xfft[0])])
    plt.axis([ xfft[2][fnz(xfft[0])]-1,xfft[2][lnz(xfft[0])]+1, min(xfft[0])-0.5, max(xfft[0])+0.5])
    plt.grid()
    fig.add_subplot(gs[2,0])
    plt.stem(xfft[2],xfft[1])
    plt.grid()
    plt.ylabel("angle X(f)")
    plt.xlabel("f [Hz]")
    fig.add_subplot(gs[2,1])
    plt.stem(xfft[2],xfft[1])
    plt.axis([ xfft[2][fnz(xfft[0])]-1,xfft[2][lnz(xfft[0])]+1, min(xfft[1])-0.5, max(xfft[1])+0.5])
    plt.grid()
    plt.xlabel("f [Hz]")

#Task 2
fs = 100
x = 5*np.sin(2*np.pi*t)
plt_usr_FFT(x,fs,0,2,"Task 2 - Second Sinusoid Signal")
x = 2*np.cos((2*np.pi*2*t)-2) + np.power(np.sin((2*np.pi*6*t)+3),2)
plt_usr_FFT(x,fs,0,2,"Task 3 - Composite Sinusoid")

#Task 3
x = np.cos(2*np.pi*t)
plt_usr_FFT2(x,fs,0,2,"Task 4.1 - Improved Plots for cos(2pi*t)")
x = 5*np.sin(2*np.pi*t)
plt_usr_FFT2(x,fs,0,2,"Task 4.2 - Improved Plots for the Second Function")
x = 2*np.cos((2*np.pi*2*t)-2) + np.power(np.sin((2*np.pi*6*t)+3),2)
plt_usr_FFT2(x,fs,0,2,"Task 4.3 - Improved Plots for the Third Function")

#Task 4
K = np.arange(1, 15+1, 1)
bk = (2/(np.pi*K))*(1-np.cos(K*np.pi))
x = 0
T = 8 #seconds
fs = (1/T)*1000.0
steps = 1.0/fs
w = (2*np.pi)/T
t = np.arange(0, 16, steps)
for i in range(1,15+1):
    x += bk[i-1]*np.sin(i*w*t)
plt_usr_FFT2(x,fs,0,16,"Task 5 - Lab 8 Square Wave (N=15)")

#https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_multicolumn.html