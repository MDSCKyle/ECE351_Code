################################################################
# Kyle McCain                                                  #
# ECE 351-53                                                   #
# LAB 11                                                       #
# Due 4/20/2021                                                #
# Z transform lab                                              #
################################################################

#%% function code
# -*- coding: utf-8 -*-
"""
@author: Phillip Hagen

Description:zplane()
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Zplane function

#
# Copyright (c) 2011 Christopher Felton
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# The following is derived from the slides presented by
# Alexander Kain for CS506/606 "Special Topics: Speech Signal Processing"
# CSLU / OHSU, Spring Term 2011.
#
#
#
# Modified by Drew Owens in Fall 2018 for use in the University of Idaho's 
# Department of Electrical and Computer Engineering Signals and Systems I Lab
# (ECE 351)
#
# Modified by Morteza Soltani in Spring 2019 for use in the ECE 351 of the U of
# I.
#
# Modified by Phillip Hagen in Fall 2019 for use in the University of Idaho's  
# Department of Electrical and Computer Engineering Signals and Systems I Lab 
# (ECE 351)
    
def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import patches    
    
    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'o', ms=10,label='Zeros')
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0)

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'x', ms=10,label='Poles')
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0)

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.legend()

    # set the ticks
    # r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    
    return z, p, k
#%% Lab code

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import control as con

#task 1 - handwrite H(z)
Hz_num = [2,-40] # 2 -40z^-1
Hz_den = [1,-10,16] # 1 -10z^-1 + 16^-2

#manual PFT: got -4/(z-8) and 6/(z-2)
r,p,k = sig.residuez(Hz_num,Hz_den)
print("r = ",r)
print("p = ",p)
print("k = ",k)
#Results agreed with hand derived calcs.

#Task 4 - use given function to generate plot with zeros and poles
z,p2,k2 = zplane(Hz_num,Hz_den)

#Task 5 - use scipy to plot linear bode plot for h(z)
w,h = sig.freqz(Hz_num,Hz_den, whole="True")
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(w,h)
plt.title("Bode Plot for H(z)")
plt.ylabel("Gain or Attenuation")
plt.grid()
angles = np.unwrap(np.angle(h))
plt.subplot(2,1,2)
plt.plot(w,angles)
plt.xlabel("rad/sec")
plt.ylabel("Phase Shift (Radians)")
plt.grid()
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.residuez.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html?highlight=freqz#scipy.signal.freqz