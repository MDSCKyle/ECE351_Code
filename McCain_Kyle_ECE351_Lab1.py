################################################################
# Kyle McCain                                                  #
# ECE351-53                                                    #
# LAB 1                                                        #
# Due 1/26/2021                                                #
# Lab used to exercise basic Python / library functionality    #
################################################################

#Use 'as' during import to avoid repetitive typing
#import numpy
import numpy as np
#import scipy.signal
import scipy.signal as sig
import time
import matplotlib.pyplot as plt

# Testing Python variables and print commands
# t = 1
# print(t)
# print("t = ",t)
# print('t =',t,"seconds")
# print('t is now =',t/3,'\n...and can be rounded using `round()`',round(t/3,4))

#output of above:
# 1
# t =  1
# t = 1 seconds
# t is now = 0.3333333333333333 
# ...and can be rounded using `round()` 0.3333


# Testing math (exponents)
#print(3**2)

#output of above:
# 9


# Numpy arrays - operate like a list
# list1 = [0,1,2,3]
# print('list1:',list1)
# list2 = [[0],[1],[2],[3]]
# print('list2:',list2)
# list3 = [[0,1],[2,3]]
# print('list3:',list3)

# array1 = numpy.array([0,1,2,3])
# print('array1:',array1)
# array2 = numpy.array([[0],[1],[2],[3]])
# print('array2:',array2)
# array3 = numpy.array([[0,1],[2,3]])
# print('array3:',array3)

#output of above:
# list1: [0, 1, 2, 3]
# list2: [[0], [1], [2], [3]]
# list3: [[0, 1], [2, 3]]
# array1: [0 1 2 3]
# array2: [[0]
#  [1]
#  [2]
#  [3]]
# array3: [[0 1]
#  [2 3]]


# Calling packages using changed import statments
# print(np.pi)

#output of above:
#3.141592653589793


#making larger arrays using numpy
# print(np.arange(4),'\n',
#       np.arange(0,2,0.5),'\n',
#       np.linspace(0,1.5,4))

#output of above:
 # [0 1 2 3] 
 # [0.  0.5 1.  1.5] 
 # [0.  0.5 1.  1.5]
 
 
#Indexing lists/arrays in python. First element is 0
# list1 = [1,2,3,4,5]
# array1 = np.array(list1) # define a numpy array using a python list
# print('list1 :',list1[0],list1[4])
# print('array1:',array1[0],array1[4])
# array2 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
# list2 = list(array2) # define a python list using a numpy array/matrix
# print('array2:',array2[0,2],array2[1,4])
# print('list2 :',list2[0][2],list2[1][4])
#numpy has better matrix indexing

#output of above (1)
# list1 : 1 5
# array1: 1 5
# array2: 3 10
# list2 : 3 10

#access entire row or col of a numpy array using :
# print(array2[:,2],array2[0,:])

#output of above (2)
# [3 8] [1 2 3 4 5]


#Define a matrix as an array of zeros or ones.
# print('1x3:',np.zeros(3))
# print('2x2',np.zeros((2,2)))
# print('2x3',np.ones((2,3)))

#output of above
# 1x3: [0. 0. 0.]
# 2x2 [[0. 0.]
#  [0. 0.]]
# 2x3 [[1. 1. 1.]
#  [1. 1. 1.]]


#Intro to matlpotlib.pyplot. See external docs on this library
#Define Variables
# steps = 0.1 #step size
# x = np.arange(-2,2+steps,steps)
# y1 = x+2
# y2 = x**2
# #plots
# plt.figure(figsize=(12,8)) #new figure of custom size 12,8
# plt.subplot(3,1,1,) #subplot 1: (row,col,number)
# plt.plot(x,y1) #(x axis, y axis) variables to plot
# plt.title('Sample Plots for Lab 1') #entire figure
# plt.ylabel('Subplot 1') #Y axis label
# plt.grid(True)
# plt.subplot(3,1,2) #subplot2
# plt.plot(x,y2)
# plt.ylabel('Subplot 2')
# plt.grid(which='both')
# plt.subplot(3,1,3)
# plt.plot(x,y1,'--r',label='y1')
# plt.plot(x,y2,'o',label='y2')
# plt.axis([-2.5, 2.5, -0.5, 4.5]) #zoom level
# plt.grid(True)
# plt.legend(loc='lower right')
# plt.xlabel('x')
# plt.ylabel('Subplot 3')
# plt.show() #render plots

#output of above was graphical.


#Complex numbers in python
# cRect = 2 + 3j
# print(cRect)
# cPol = abs(cRect) * np.exp(1j*np.angle(cRect))
# print(cPol) # output will be rectangular
# cRect2 = np.real(cPol) + 1j*np.imag(cPol)
# print(cRect2) #converting polar to recangular.

#ouput of above:
# (2+3j)
# (2+2.9999999999999996j)
# (2+2.9999999999999996j)


#Issues with nan (not a number)
# print(np.sqrt(3*5 - 5*5)) #fail
# print(np.sqrt(3*5 - 5*5 + 0j)) #output 3.1622776601683795j


#List of all packages used this semester. Using as is optional. Not all labs
#   need to use all libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy as sp
# import scipy.signal as sig
# import pandas as pd
# import control
# import time
# from scipy.fftpack import fft, fftshift


#Useful python commands - pupose described
# range() #creates a range of numbers. good in 'for' loops
# np.arange()#np array. step size can be a decimal.
# np.append()#add more values to the end of an np array
# np.insert()#add values to the start of an np array
# np.concatenate()#combine np arrays
# np.linspace()#np array with linear range of values and X elements
# np.logspace()#np array with log base X range, Y elements
# np.reshape()#reshapes np array without deleting data. (numpy)
# np.transpose()#change dimmensions of array (numpy)
# len()#returns the horizontal number of array elements
# somearray.size#returns the vertical number of array elements
# somearray.shape#returns the dimmensions of the array
# somearray.reshape#reshape the dimensions of an array (python)
# somearray.T#transpose array function (python)