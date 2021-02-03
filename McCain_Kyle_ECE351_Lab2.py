################################################################
# Kyle McCain                                                  #
# ECE351-53                                                    #
# LAB 2                                                        #
# Due 2/2/2021                                                 #
# Lab to learn python functions - more numpy - step functions  #
################################################################

import numpy as np
import scipy.signal as sig
import time
import matplotlib.pyplot as plt

#%%PART 1
#Following the example code from part 3.0.1, write a user defined function that plots y=cos(t)
#Then plot it from 0 <= t <= 10 seconds.
plt.rcParams.update({'font.size':14})

steps = 1e-3
t = np.arange(0, 10+steps, steps)

#User defined functiuon for y(t) = cos(t)
def usrcos(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        y[i] = np.cos(t[i])
    return y

#Get output values and graph
y = usrcos(t)
plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('User cos function output')
plt.title('PART 1 - TASK 2 - First User Defined Function')




#%%PART 2
#Equation found for figure 2 in lab handout - will require LaTeX formatting.
#Need to make two user defined functions - one for unit step, one for ramp.
#Use usr equations to recreate figure 2 from -5 to 10 seconds


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

#Testing new functions - if steps is set to a larger number, the derivative plot breaks
steps = 1e-3
t = np.arange(-5, 10+steps, steps)

#new figure
plt.figure(figsize = (10,7))

#plot the unit step function
y = u(t)
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('Unit Step Function')
plt.title('PART 2 - TASK 2 - Unit Step and Ramp Functions')

#plot the ramp function
y = r(t)
plt.subplot(2,1,2)
plt.plot(t, y)
plt.grid()
plt.ylabel('Ramp Function')

#Function that generates figure 2.
def f2(t):
    y = r(t) - r(t-3) + 5*u(t-3) - u(t-6) - 2*r(t-6)
    return y

#new figure for the handwritten equation to match figure 2
plt.figure(figsize = (15,15))
#fig2 = r(t) - r(t-3) + 5*u(t-3) - u(t-6) - 2*r(t-6) #the objective was to match figure 2 using a user defined function, but this did work to generate the figure at least until part 3
y = f2(t)
plt.subplot(2,2,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.xticks(np.arange(-5,11,1)) #These commands correct the gridlines. Helps to check accuracy.
plt.yticks(np.arange(-3,10,1))
plt.title('PART 2 - TASK 3 - Matching Figure 2')



#%%PART 3
#Using fig2 from before:
    #apply a time reversal - plot
    #apply a time shift operations f(t-4) and f(-t-4)
    #Time scale by 2 and 1/2
    #Plot the derivative by hand (not code)
    #use numpy.diff() and plot its effect on fig2.

#Time reversal.
plt.figure(figsize = (15,15))
t = np.arange(-10, 5+steps, steps)
y = f2(-t)
plt.subplot(2,2,1)
plt.axis([-11,6,-3,9])
plt.plot(t,y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f(-t)')
plt.title('Time Reversal')

#Time-Shift.
plt.figure(figsize = (10,7))
t = np.arange(-20, 15+steps, steps)
y = f2(t-4)
plt.subplot(2,1,1)
plt.axis([-2,15,-3,9])
plt.plot(t,y)
plt.title('Time Shift Operations')
plt.grid()
plt.ylabel('f(t-4)')
y = f2(-t-4)
plt.subplot(2,1,2)
plt.axis([-15,0,-3,9])
plt.plot(t,y)
plt.grid()
plt.ylabel('f(-t-4)')
plt.xlabel('t')

#Time-Scale.
plt.figure(figsize = (10,7))
t = np.arange(-10, 30+steps, steps)
y = f2(t/2)
plt.subplot(2,1,1)
plt.axis([-4,25,-3,9])
plt.plot(t,y)
plt.title('Time Scale Operations')
plt.grid()
plt.ylabel('f(t/2)')
y = f2(2*t)
plt.subplot(2,1,2)
plt.axis([-1,6,-3,9])
plt.plot(t,y)
plt.grid()
plt.ylabel('f(2t)')
plt.xlabel('t')

#Derivative using numpy.diff
plt.figure(figsize = (15,15))
t = np.arange(-5, 10+steps, steps)
dt = np.diff(t) #small change in time (basically just an array with all values = steps)
y = f2(t)
dy = np.diff(y,axis=0) #array of values that show the magnitude of change between y(t) and y(t+1)
yprime = dy/dt #should approximate derivative if step size is small enough
plt.subplot(2,2,1)
plt.axis([-5,10,-3,9])
plt.plot(t[range(len(yprime))],yprime)
plt.title('Derivative')
plt.grid()
plt.ylabel('df(t)/dt')
plt.xlabel('t')







#%%Example Code from part 3.0.1
# plt.rcParams.update({'font.size': 14})

# steps = 1e-2 #Defines the step size
# t = np.arange(0, 5 + steps, steps) #from 0 to 5 (inclusive)

# print('Number of elements in t: len(t) = ', len(t), '\nFirst Element: t[0]', t[0], '\nLast Element: t[len(t)-1] = ', t[len(t)-1])
# #Len returns the num. elements in an array, but the max index is one less than this.

# # Example of a user defined function:
# # Create output y(t) using a for loop and if/else statements as an array of output values that come from a structurally identical input array
# def example1(t):
#     y = np.zeros(t.shape)
    
#     for i in range(len(t)): #for loop will iterate through each element of array t
#         if i < (len(t) + 1)/3:
#             y[i] = t[i]**2
#         else:
#             y[i] = np.sin(5*t[i]) + 2
#     return y

# y = example1(t)

# #plot the example:
# plt.figure(figsize = (10, 7))
# plt.subplot(2,1,1)
# plt.plot(t, y)
# plt.grid()
# plt.ylabel('y(t) with Good Resolution')
# plt.title('Background - Illustration of for Loops and if/else Statements')

# t = np.arange(0, 5 + 0.25, 0.25) #new set of input values - poor resolution
# y = example1(t)

# plt.subplot(2,1,2)
# plt.plot(t, y)
# plt.grid()
# plt.ylabel('y(t) with Poor Resolution')
# plt.xlabel('t')
# plt.show()
