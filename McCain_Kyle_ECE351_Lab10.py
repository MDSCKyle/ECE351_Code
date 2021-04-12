################################################################
# Kyle McCain                                                  #
# ECE 351-53                                                    #
# LAB 10                                                       #
# Due 4/13/2021                                                #
# Multiple Method Bode Plot Generation using Python & Libraries#
################################################################

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import control as con

#Task 1 - Manual Bode Plot
steps = 1e3
R = 1e3 #ohms
L = 27e-3 #H
C = 100e-9 #F
w = np.arange(1e3, (1e6)+steps, steps)

com_term_1 = w/(R*C)
com_term_2 = ( (1/(L*C)) - w**2)

Hsmag = com_term_1 / np.sqrt( np.power(com_term_1,2)+np.power(com_term_2,2) )
#Hsmag = (w/(R*C))/(np.sqrt(w**4 + (1/(R*C)**2 - 2/(L*C))*w**2 + (1/(L*C))**2))
Hsmag_db = 20*np.log10(Hsmag)
#LOG != LOG10

Hsang = (np.pi/2) - np.arctan2(com_term_1,com_term_2)
Hsang = Hsang*180/np.pi

#Plot for Task 1
plt.figure(figsize = (10,6))
plt.subplot(2,1,1)
plt.semilogx(w,Hsmag_db)
plt.ylabel("Magnitude (dB)")
plt.title("Bode Plot 1 - Manual Magnitude/Phase Expressions")
plt.grid(which = "both", ls = '-')
plt.subplot(2,1,2)
plt.semilogx(w,Hsang)
plt.ylabel("Phase (deg)")
plt.xlabel("Frequency (rad/sec)")
plt.grid(which = "both", ls = '-')

#Task 2 - scipy
ct1 = 1/(R*C)
ct2 = 1/(L*C)
w2,Hsmag2,Hsang2 = sig.bode(([ct1,0],[1,ct1,ct2]),w)
plt.figure(figsize = (10,6))
plt.subplot(2,1,1)
plt.semilogx(w2,Hsmag2)
plt.ylabel("Magnitude (dB)")
plt.title("Bode Plot 2 - Scipy.Signal.Bode() Plot")
plt.grid(which = "both", ls = '-')
plt.subplot(2,1,2)
plt.semilogx(w2,Hsang2)
plt.ylabel("Phase (deg)")
plt.xlabel("Frequency (rad/sec)")
plt.grid(which = "both", ls = '-')


#Task 3 - control
sys = con.TransferFunction([ct1,0],[1,ct1,ct2])
_ = con.bode(sys, w, dB=True, Hz = True, deg = True, plot = True)


#%%part 2

fs = 5e5
steps = 1/fs
t = np.arange(0, (1e-2)+steps, steps)
x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)
plt.figure(figsize = (10,6))
plt.plot(t,x)
plt.grid()
plt.title("Composite Signal x(t) - Unfiltered")
plt.ylabel("Amplitude (Volts)")
plt.ylabel("t (sec)")
z,p = sig.bilinear([ct1,0],[1,ct1,ct2],fs)
y = sig.lfilter(z,p,x)
plt.figure(figsize = (10,6))
plt.plot(t,y)
plt.grid()
plt.title("Composite Signal x(t) - Filtered")
plt.ylabel("Amplitude (Volts)")
plt.ylabel("t (sec)")