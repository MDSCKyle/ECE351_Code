################################################################
# Kyle McCain                                                  #
# ECE351-53                                                    #
# LAB 3                                                        #
# Due 2/9/2021                                                 #
# User defined convolution function.                           #
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


steps = 0.01
t = np.arange(0, 20+steps, steps)

#%% PART 1 - Getting functions to convolve.
f1 = u(t-2) - u(t-9)
f2 = np.exp(-t)*u(t)
f3 = r(t-2) * (u(t-2) - u(t-3)) + r(4-t) * (u(t-3) - u(t-4))

plt.figure(figsize = (10,10))
plt.subplot(3,1,1)
plt.plot(t,f1)
plt.grid()
plt.ylabel("f1(t)")
plt.axis([0, 10, -0.1, 1.1]) #zoom level
plt.title("Plotting Task 1 Functions")
plt.subplot(3,1,2)
plt.plot(t,f2)
plt.grid()
plt.ylabel("f2(t)")
plt.axis([0, 7, -0.1, 1.1]) #zoom level
plt.subplot(3,1,3)
plt.plot(t,f3)
plt.grid()
plt.ylabel("f3(t)")
plt.xlabel("t")
plt.axis([0, 6, -0.1, 1.1]) #zoom level


#%% PART 2 - User defined convolve function
#user defined convolution.
#INPUT - three equally sized arrays of numbers (2 f(t) outputs and 1 for t)
#OUTPUT - a tuple where out[0] is the function output and out[1] is new t
def myconv(y,x,t):
    steps = t[1] - t[0] #get time resolution
    t = np.arange(0,(2*t[len(t)-1])+steps, steps) #get new time array
    z = np.zeros((len(y)+len(x))-1) #define convolution output array
    if len(t) != len(z): #truncate t if it is larger than z.
        t = t[0:len(t)-(len(t)-len(z))] #...done because t is floating point
    for i in range(len(x)): #for every num in second function,
        for j in range(len(y)): #for every num in first function,
            if (i+j+1 < len(z)): #add 1 step offset for better accuracy
                z[i+j+1] += (x[i]*y[j])*steps #perfrom convolution
    return (z,t) #return tuple with convolution array and new t array

##F1 CONV F2
plt.figure(figsize = (15,10))
x = myconv(f1,f2,t)
plt.subplot(2,1,1)
plt.plot(x[1], x[0])
plt.grid()
plt.ylabel("myconv")
plt.axis([0, 16, -0.1, 1.1]) #zoom level
plt.title('f1(t) Convolved With f2(t)')
plt.subplot(2,1,2)
plt.plot(x[1],sig.convolve(f1,f2)*steps)
plt.grid()
plt.ylabel("scipy.sig conv")
plt.xlabel("t")
plt.axis([0, 16, -0.1, 1.1]) #zoom level

#F2 CONV F3
plt.figure(figsize = (15,10))
x = myconv(f2,f3,t)
plt.subplot(2,1,1)
plt.plot(x[1], x[0])
plt.grid()
plt.ylabel("myconv")
plt.axis([0, 11, -0.1, 0.6]) #zoom level
plt.title('f2(t) Convolved With f3(t)')
plt.subplot(2,1,2)
plt.plot(x[1],sig.convolve(f2,f3)*steps)
plt.grid()
plt.ylabel("scipy.sig conv")
plt.xlabel("t")
plt.axis([0, 11, -0.1, 0.6]) #zoom level

#F1 CONV F3
plt.figure(figsize = (15,10))
x = myconv(f1,f3,t)
plt.subplot(2,1,1)
plt.plot(x[1], x[0])
plt.grid()
plt.ylabel("myconv")
plt.axis([0, 16, -0.1, 1.1]) #zoom level
plt.title('f1(t) Convolved With f3(t)')
plt.subplot(2,1,2)
plt.plot(x[1],sig.convolve(f1,f3)*steps)
plt.grid()
plt.ylabel("scipy.sig conv")
plt.xlabel("t")
plt.axis([0, 16, -0.1, 1.1]) #zoom level