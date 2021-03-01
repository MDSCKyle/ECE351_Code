################################################################
# Kyle McCain                                                  #
# ECE351-53                                                    #
# LAB 5                                                        #
# Due 2/16/2021                                                #
# Step and impulse Response Lab RLC Circuit Analysis           #
################################################################

import numpy as np
import scipy.signal as sig
import time
import matplotlib.pyplot as plt


steps = 0.0000001 #needs to be small because 1 time step is one millisecond
t = np.arange(0, 1.2e-3+steps, steps)
R = 1000 #ohms
L = 27e-3 #Henry
C = 100e-9 #Farads


#Manually calculated h(t)
B = (1/(R*C)) #First Shared term in H(s)
C = (1/(L*C)) #Second shared term
#print("b=",B," c=",C)
p = (-B + np.sqrt( np.power(B,2) - (4*C) + (0*1j)))/2
#print("p=",p)
g = B*p
gmag = np.abs(g)
gang = np.angle(g) #in radians
alpha = np.real(p)
w = np.abs(np.imag(p))
htman = (gmag/w)*np.exp(alpha*t)*np.sin((w*t) + gang)

#auto h(t) using scipy signal
HSnum = [B, 0]
HSden = [1, B, C]
tout,htauto = sig.impulse((HSnum,HSden), T=t)

#Figure 1 - Plotting h(t), scipy vs hand calc
plt.figure(figsize = (10,10))
plt.subplot(2,1,1)
plt.plot(t,htman)
plt.grid()
plt.ylabel("h(t) from prelab")
# #plt.axis([0, 10, -0.1, 1.1]) #zoom level
plt.title("Task 1 - H(t) plots")
plt.subplot(2,1,2)
plt.plot(tout,htauto)
plt.grid()
plt.ylabel("h(t) by scipy")
plt.xlabel("t")

#signal step - task 2. Plots system step response.
tout2,hstep = sig.step((HSnum,HSden), T=t)
plt.figure(figsize = (8,6))
plt.plot(tout2,hstep)
plt.grid()
plt.ylabel("H(s) step reponse")
plt.xlabel("t")
