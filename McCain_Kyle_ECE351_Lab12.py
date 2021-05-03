################################################################
# Kyle McCain                                                  #
# ECE 351-53                                                   #
# LAB 12                                                       #
# Due 5/4/2021                                                 #
# Final Lab Project - Noisy Signal Filtering                   #
################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd
import scipy.fftpack as ft

#FFT function to find composite freqeuncies
def usr_FFT(x,fs):
    N = len(x)
    X_fft = ft.fft(x)
    X_fft_shifted = ft.fftshift(X_fft)
    freq = np.arange(-N/2,N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return (X_mag,X_phi,freq)

#faster stem function
def make_stem(ax,x,y,color='k',style='solid',label='',linewidths=2.5,**kwargs):
    ax.axhline(x[0],x[-1],0, color='r')
    ax.vlines(x,0,y,color=color,linestyles=style,label=label,linewidths=linewidths)
    ax.set_ylim([1.05*y.min(),1.05*y.max()])


#The following code generates a sereis of plots that demonstrate the engineer-
#ing problem and solution.
#PART 1 - Characterization of the noisy signal using plotting and FFT.
#PART 2 - Designing a filter that meets a list of requirements.
#PART 3 - Showing the effects of the filter on the signal.
#PART 4 - Comparing the signal before and after filtration.


#%%PART 1
#This section of code loads and plots the noisy waveform stored in a CSV file.
#A fast fourier transform is used to determine the amplitude and frequency of
#every sine wave the makes up the total signal. This helps to identify the key
#frequencies that need to be attenuated or maintained. Phase shifting is not
#a concern for this project.

#load the noisy signal using the Pandas Library
df = pd.read_csv('NoisySignal.csv')
fs = 1e6 #This value was given in the file.

t = df['0'].values
sensor_sig = df['1'].values

#plot the raw signal
plt.figure(figsize =(10,7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noisy Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [v]')
plt.show()

#plot frequencies using FFT (The IF statement was added to tweak plots that
#occur after this point, since the make_stem function takes some time to process.)
nostem = 1;
if nostem == 1:
    fs = 1e6
    nsmag,nsang,atfreq = usr_FFT(sensor_sig,fs)
    fig, ax1 = plt.subplots(figsize=(10,7))
    plt.subplot(ax1)
    make_stem(ax1, atfreq, nsmag)
    plt.xscale('log')
    plt.xlim([10,1e6])
    plt.grid(which = 'both')
    plt.title("Noisy Signal - All Component Frequencies")
    plt.xlabel("f [hz]")
    plt.ylabel("Amplitude")
    plt.show()
    
    #zoom into key areas of interest (same process as before, but with x/ylims)
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,7))
    plt.subplot(ax1)
    make_stem(ax1, atfreq, nsmag)
    plt.xlim([10,1e3])
    plt.grid()
    plt.title("Noisy Signal - Key Frequencies")
    plt.ylabel("Low Freq. to Filter")
    plt.subplot(ax2)
    make_stem(ax2, atfreq, nsmag)
    plt.xlim([1.05e3,2.7e3])
    plt.grid()
    plt.ylabel("Sensor Data Freq. to Keep")
    plt.subplot(ax3)
    make_stem(ax3, atfreq, nsmag)
    plt.xscale('log')
    plt.xlim([5e3,4e5])
    plt.ylim([0,0.8])
    plt.grid(which = "both")
    plt.ylabel("High Freq. to Filter")
    plt.xlabel("f [hz]")



#%%PART 2
#This part of the lab was mostly done using pen and paper design/calculations.
#What appears here is a final check of the design using python functions
#explored in previous lab assignments.
#There are infinitely many filter designs, but a basic RLC bandpass filter was
#chosen since the transfer function had already been derivied. In addition,
#it was easy to find equations for the center frequency and quality factor.
#Once a decent enough filter was found using realistic components, the values
#were loaded into python and a bode plot was generated. Parts of this plot
#are carefully disected to verify all design requirements are met.


#Bode plot of filter:
#Standard Values (L) : http://www.rfcafe.com/references/electrical/inductor-values.htm#:~:text=Standard%20Inductor%20Values%20%20%201.0%20%20,%20%201300%20%2022%20more%20rows%20
#Standard Values (R,C) : James W. Nilsson - Electric Circuits - Appendix H
#These values were determined to be the best fit based on center frequency
#and quality factor calculations. In addition, they consist of components
#that can be found through distributors like Mouser or DigiKey.
steps = 1e2
R = 2000 #ohms
L = 68e-3 #H
C = 0.1033e-6 #F
w = np.arange(2*np.pi*10, 2*np.pi*1e6+steps, steps)
ct1 = 1/(R*C)
ct2 = 1/(L*C)

#The Scipy Signal Bode function uses the transfer function to generate the
#bode plot.
w2,Hsmag2,Hsang2 = sig.bode(([ct1,0],[1,ct1,ct2]),w)
#complete bode plot
plt.figure(figsize = (10,6))
plt.subplot(2,1,1)
plt.semilogx(w2/(2*np.pi),Hsmag2)
plt.ylabel("Magnitude (dB)")
plt.title("RLC Filter Bode Plot - Complete")
plt.grid(which = "both", ls = '-')
plt.subplot(2,1,2)
plt.semilogx(w2/(2*np.pi),Hsang2)
plt.ylabel("Phase (deg)")
plt.xlabel("Frequency [Hz]")
plt.grid(which = "both", ls = '-')
#zoom in to desired frequencies
# 1800hz and 2kHz frequencies cannot be attenuated more than -0.3dB
plt.figure(figsize = (10,6))
plt.subplot(2,1,1)
plt.semilogx(w2/(2*np.pi),Hsmag2)
plt.axhline(y = -0.3, color = 'r')
plt.axvline(x = (1800), color = 'g')
plt.axvline(x = (2000), color = 'g')
plt.xlim([1.7e3,2.1e3])
plt.ylim([-1.1,0.1])
plt.ylabel("Magnitude (dB)")
plt.title("RLC Filter Bode Plot - Attenuation of Desired Frequencies")
plt.grid(which = "both", ls = '-')
plt.subplot(2,1,2)
plt.semilogx(w2/(2*np.pi),Hsang2)
plt.xlim([1.7e3,2.1e3])
plt.ylim([-35,35])
plt.ylabel("Phase (deg)")
plt.xlabel("Frequency [Hz]")
plt.grid(which = "both", ls = '-')
#zoom in to show low frequency attenuation
# The 60Hz noise must be attenuated by at least -30 dB.
plt.figure(figsize = (10,6))
plt.subplot(2,1,1)
plt.semilogx(w2/(2*np.pi),Hsmag2)
plt.axhline(y = -30, color = 'r')
plt.axvline(x = (60), color = 'g')
plt.xlim([10,250])
plt.ylim([-55,-25])
plt.ylabel("Magnitude (dB)")
plt.title("RLC Filter Bode Plot - Attenuation of Low Frequencies")
plt.grid(which = "both", ls = '-')
plt.subplot(2,1,2)
plt.semilogx(w2/(2*np.pi),Hsang2)
plt.xlim([10,200])
plt.ylim([85,95])
plt.ylabel("Phase (deg)")
plt.xlabel("Frequency [Hz]")
plt.grid(which = "both", ls = '-')
#zoom in to show high frequency attenuation
#  All frequencies greater than 50kHz must be attenuated by at least -21dB.
plt.figure(figsize = (10,6))
plt.subplot(2,1,1)
plt.semilogx(w2/(2*np.pi),Hsmag2)
plt.axhline(y = -21, color = 'r')
plt.axvline(x = 50e3, color = 'g')
plt.xlim([5e3,1e6])
plt.ylim([-70,-10])
plt.ylabel("Magnitude (dB)")
plt.title("RLC Filter Bode Plot - Attenuation of High Frequencies")
plt.grid(which = "both", ls = '-')
plt.subplot(2,1,2)
plt.semilogx(w2/(2*np.pi),Hsang2)
plt.xlim([5e3,1e6])
plt.ylim([-92,-78])
plt.ylabel("Phase (deg)")
plt.xlabel("Frequency [Hz]")
plt.grid(which = "both", ls = '-')


#%%PART 3
#Part 3 is essentially part 1, but with each plot replaced with the filtred
#output. Scipy functions bilinear and lfilter are used to generate these.


#Perform filtering using Scipy Signal BILINEAR and Lfilter
z,p = sig.bilinear([ct1,0],[1,ct1,ct2],fs)
sensor_filtered = sig.lfilter(z,p,sensor_sig)
plt.figure(figsize = (10,7))
plt.plot(t,sensor_filtered)
plt.grid()
plt.title('Filtered Output Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [v]')

#Redo the stem plot
if nostem == 1:
    nsmag_filtered,nsang_filtered,atfreq_filtered = usr_FFT(sensor_filtered,fs)
    fig, ax1 = plt.subplots(figsize=(10,7))
    plt.subplot(ax1)
    make_stem(ax1, atfreq_filtered, nsmag_filtered)
    
    plt.xscale('log')
    plt.xlim([10,1e6])
    plt.grid(which = 'both')
    plt.title("Filtered Signal - All Component Frequencies")
    plt.xlabel("f [hz]")
    plt.ylabel("Amplitude")
    plt.show()
    
    #zoom into key areas
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,7))
    plt.subplot(ax1)
    make_stem(ax1, atfreq_filtered, nsmag_filtered)
    plt.xlim([10,1e3])
    plt.grid()
    plt.title("Filtered Signal - Key Frequencies")
    plt.ylabel("Low Freq. After Filter")
    plt.subplot(ax2)
    make_stem(ax2, atfreq_filtered, nsmag_filtered)
    plt.xlim([1.05e3,2.7e3])
    plt.grid()
    plt.ylabel("Sensor Frequencies")
    plt.subplot(ax3)
    make_stem(ax3, atfreq_filtered, nsmag_filtered)
    plt.xscale('log')
    plt.xlim([5e3,4e5])
    plt.ylim([0,0.8])
    plt.grid(which = "both")
    plt.ylabel("High Freq. After Filter")
    plt.xlabel("f [hz]")
    
    #Final Check - show that 1e5 hz frequencies are all less than 0.05
    fig, ax1 = plt.subplots(figsize=(10,7))
    plt.subplot(ax1)
    make_stem(ax1, atfreq_filtered, nsmag_filtered)
    plt.xscale('log')
    plt.xlim([50e3,1e6])
    plt.ylim([0.0,0.06])
    plt.axhline(y = 0.05, color = 'r')
    plt.axvline(x = 1e5, color = 'g')
    plt.grid(which = "both")
    plt.title("Complete Attenuation of Freq > 100kHz:")
    plt.ylabel("Magnitude [v]")
    plt.xlabel("f [hz]")
    
    
#%% PART 4
#A combination of parts 1 and 3. Both stem plots are rendered on the same figure
#for a quick visual comparison.    
    
    #Comparison plots between filtered and unfiltered FFT
    fig, ax1 = plt.subplots(figsize=(10,7))
    plt.subplot(ax1)
    make_stem(ax1, atfreq,nsmag, color='b')
    make_stem(ax1, atfreq_filtered, nsmag_filtered)
    
    plt.xscale('log')
    plt.xlim([10,1e6])
    plt.grid(which = 'both')
    plt.title("Signal Comparision - Blue is Unfiltered")
    plt.xlabel("f [hz]")
    plt.ylabel("Amplitude")
    plt.show()
    
    #zoom into key areas
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,7))
    plt.subplot(ax1)
    make_stem(ax1, atfreq,nsmag, color='b')
    make_stem(ax1, atfreq_filtered, nsmag_filtered)
    plt.xlim([10,1e3])
    plt.grid()
    plt.title("Signal Comparison - Key Frequencies - Blue is Unfiltered")
    plt.ylabel("Low Freq.")
    plt.subplot(ax2)
    make_stem(ax2, atfreq,nsmag, color='b')
    make_stem(ax2, atfreq_filtered, nsmag_filtered)
    plt.xlim([1.05e3,2.7e3])
    plt.grid()
    plt.ylabel("Desired Freq.")
    plt.subplot(ax3)
    make_stem(ax3, atfreq,nsmag, color='b')
    make_stem(ax3, atfreq_filtered, nsmag_filtered)
    plt.xscale('log')
    plt.xlim([5e3,4e5])
    plt.ylim([0,0.8])
    plt.grid(which = "both")
    plt.ylabel("High Freq.")
    plt.xlabel("f [hz]")