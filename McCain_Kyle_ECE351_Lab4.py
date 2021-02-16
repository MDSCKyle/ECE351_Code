################################################################
# Kyle McCain                                                  #
# ECE351-53                                                    #
# LAB 4                                                        #
# Due 2/16/2021                                                #
# Step Response Lab                                            #
################################################################

import numpy as np
import scipy.signal as sig
import time
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

#ramp function
#0 if input is less than 0. equal to input if input >=1
def r(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

#user defined convolution.
#INPUT - three equally sized arrays of numbers (2 f(t) outputs and 1 for t)
#OUTPUT - a tuple where out[0] is the function output and out[1] is new t
def myconv(y,x,t):
    steps = t[1] - t[0] #get time resolution
    t = np.arange(2*t[0],(2*t[len(t)-1])+steps, steps) #get new time array
    z = np.zeros((len(y)+len(x))-1) #define convolution output array
    if len(t) != len(z): #truncate t if it is larger than z.
        t = t[0:len(t)-(len(t)-len(z))] #...done because t is floating point
    for i in range(len(x)): #for every num in second function,
        for j in range(len(y)): #for every num in first function,
            if (i+j+1 < len(z)): #add 1 step offset for better accuracy
                z[i+j+1] += (x[i]*y[j])*steps #perfrom convolution
    return (z,t) #return tuple with convolution array and new t array

steps = 0.01
t = np.arange(-10, 10+steps, steps)

h1 = np.exp(2*t)*u(1-t)
h2 = u(t-2) - u(t-6)
h3 = np.cos(2*np.pi*0.25*t)*u(t)

plt.figure(figsize = (10,10))
plt.subplot(3,1,1)
plt.plot(t,h1)
plt.grid()
plt.ylabel("h1(t)")
#plt.axis([0, 10, -0.1, 1.1]) #zoom level
plt.title("Plotting Task 1 Functions")
plt.subplot(3,1,2)
plt.plot(t,h2)
plt.grid()
plt.ylabel("h2(t)")
#plt.axis([0, 7, -0.1, 1.1]) #zoom level
plt.subplot(3,1,3)
plt.plot(t,h3)
plt.grid()
plt.ylabel("h3(t)")
plt.xlabel("t")
#plt.axis([0, 6, -0.1, 1.1]) #zoom level

#%%Step Response myconv
y1 = myconv(h1,u(t),t)
y2 = myconv(h2,u(t),t)
y3 = myconv(h3,u(t),t)

plt.figure(figsize = (10,10))
plt.subplot(3,1,1)
plt.plot(y1[1],y1[0])
plt.grid()
plt.ylabel("y1(t)")
#plt.axis([0, 10, -0.1, 1.1]) #zoom level
plt.title("Plotting With Python MyConv")
plt.subplot(3,1,2)
plt.plot(y2[1],y2[0])
plt.grid()
plt.ylabel("y2(t)")
#plt.axis([0, 7, -0.1, 1.1]) #zoom level
plt.subplot(3,1,3)
plt.plot(y3[1],y3[0])
plt.grid()
plt.ylabel("y3(t)")
plt.xlabel("t")
#plt.axis([0, 6, -0.1, 1.1]) #zoom level

#%%Step Response scipy signal
sy1 = sig.convolve(h1,u(t))*steps
sy2 = sig.convolve(h2,u(t))*steps
sy3 = sig.convolve(h3,u(t))*steps

plt.figure(figsize = (10,10))
plt.subplot(3,1,1)
plt.plot(y1[1],sy1)
plt.grid()
plt.ylabel("y1(t)")
#plt.axis([0, 10, -0.1, 1.1]) #zoom level
plt.title("Plotting With SciPy.Signal Convolve")
plt.subplot(3,1,2)
plt.plot(y2[1],sy2)
plt.grid()
plt.ylabel("y2(t)")
#plt.axis([0, 7, -0.1, 1.1]) #zoom level
plt.subplot(3,1,3)
plt.plot(y3[1],sy3)
plt.grid()
plt.ylabel("y3(t)")
plt.xlabel("t")
#plt.axis([0, 6, -0.1, 1.1]) #zoom level

#%%Step response (manual)
tm = y1[1]
#y1m = np.exp(2*tm) + (0.5*np.exp(2*tm) - 0.5)*u(tm) - (0.5*np.exp(2*tm) - 0.5)*u(tm-1)
y1m = 0.5*np.exp(2*tm)*u(1-tm) + 0.5*np.exp(2)*u(tm-1)
y2m = r(tm-2) - r(tm-6)
#y2m = (tm-2)*u(tm-2) - (tm-6)*u(tm-6) does the same thing
y3m = (1/(2*0.25*np.pi))*np.sin((2*0.25*np.pi)*tm)*u(tm)

plt.figure(figsize = (10,10))
plt.subplot(3,1,1)
plt.plot(tm,y1m)
plt.grid()
plt.ylabel("y1(t)")
#plt.axis([0, 10, -0.1, 1.1]) #zoom level
plt.title("Plotting With Manual Convolve")
plt.subplot(3,1,2)
plt.plot(tm,y2m)
plt.grid()
plt.ylabel("y2(t)")
#plt.axis([0, 7, -0.1, 1.1]) #zoom level
plt.subplot(3,1,3)
plt.plot(tm,y3m)
plt.grid()
plt.ylabel("y3(t)")
plt.xlabel("t")
#plt.axis([0, 6, -0.1, 1.1]) #zoom level