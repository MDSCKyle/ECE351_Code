################################################################
# Kyle McCain                                                  #
# ECE351-53                                                    #
# LAB 8                                                        #
# Due 3/30/2021                                                #
# Fourier Series Square Wave Approximation                     #
################################################################

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


#Task 1 - expressions for ak, bk
K = np.arange(1, 3+1, 1)
a0 = 0
ak = 0
bk = (2/(np.pi*K))*(1-np.cos(K*np.pi))
print(bk) #requirement 1


steps = 0.001
t = np.arange(0, 20+steps, steps)
plt.figure(figsize = (5,6))
N = [1,3,15,50,150,1500]
T = 8
w = (2*np.pi)/T
pltidx = 0
for j in N:
    pltidx += 1
    if N.index(j) == 3:
        plt.xlabel("t")
        plt.figure(figsize = (5,6))
        pltidx = 1
    K = np.arange(1, j+1, 1)
    #print(K)
    a0 = 0
    ak = 0
    bk = (2/(np.pi*K))*(1-np.cos(K*np.pi))
    x = 0.5*a0
    for i in range(1,j+1):
        x += bk[i-1]*np.sin(i*w*t)

    plt.subplot(int(len(N)/2),1,pltidx)
    plt.plot(t,x)
    plt.ylabel("N = "+str(j))
    if pltidx == 1:
        if N.index(j) == 0:
            plt.title("N = 1,3,15")
        else:
            plt.title("N = 50,150,1500")
    plt.grid()

plt.xlabel("t")