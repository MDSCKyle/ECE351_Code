################################################################
# Kyle McCain                                                  #
# ECE351-53                                                    #
# LAB 6                                                        #
# Due 3/9/2021                                                 #
# Scipy Residue function & Partial Fraction Expansion          #
################################################################

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

#unit step function
#0 if input less than 0. 1 if input >= 1
def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

steps = 0.001
t = np.arange(0, 2+steps, steps)


#Task 1 - Plot h(t) (step response) from prelab
htman = 0.5 - 0.5*np.exp(-4*t) + np.exp(-6*t)

HSnum = [1, 6, 12]
HSden = [1, 10, 24]
tout,htauto = sig.step((HSnum,HSden), T=t)

HSnum2 = [1, 6, 12]
HSden2 = [1, 10, 24, 0]
R,P,_ = sig.residue(HSnum2,HSden2)
print("R=",R,"  P=",P)
ytacc = 0

for i in range(len(P)):
    alpha = np.real(P[i])
    omega = np.imag(P[i])
    kmag =  np.abs(R[i])
    kang =  np.angle(R[i]) #in radians
    ytacc += kmag*np.exp(alpha*t)*np.cos(omega*t + kang)*u(t)

#Figure 1 - Plotting h(t), scipy vs hand calc
plt.figure(figsize = (10,10))
plt.subplot(3,1,1)
plt.plot(t,ytacc)
plt.grid()
plt.ylabel("y(t) Scipy Residue + Cosine Method")
plt.title("Task 1 - y(t) Step Response Plots")
plt.subplot(3,1,2)
plt.plot(t,htman)
plt.grid()
plt.ylabel("y(t) Hand Calc.")
# #plt.axis([0, 10, -0.1, 1.1]) #zoom level
plt.subplot(3,1,3)
plt.plot(tout,htauto)
plt.grid()
plt.ylabel("y(t) by SciPy")
plt.xlabel("t")


#task 2 - plot h(t) step response for a complex differential equation
steps = 0.001
t = np.arange(0, 4.5+steps, steps)

HSnum3 = [25250]
HSden3 = [1, 18, 218, 2036, 9085, 25250, 0]
R,P,_ = sig.residue(HSnum3,HSden3)
print("R=",R,"  P=",P)
ytacc = 0

for i in range(len(P)):
    alpha = np.real(P[i])
    omega = np.imag(P[i])
    kmag =  np.abs(R[i])
    kang =  np.angle(R[i]) #in radians
    ytacc += kmag*np.exp(alpha*t)*np.cos(omega*t + kang)*u(t)


plt.figure(figsize = (8,5))
plt.plot(t,ytacc)
plt.grid()
plt.ylabel("y(t) Step Response From Part 2")
plt.xlabel("t")
plt.title("Task 2 - Step Response of a Complex System")
