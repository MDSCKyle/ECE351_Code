################################################################
# Kyle McCain                                                  #
# ECE351-53                                                    #
# LAB 7                                                        #
# Due 3/23/2021                                                #
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


#Task 1 - Block Diagrams - Open Loop
#G(s) = (s+9) / (s+4)(s+2)(s-8)
num = [1,9]
den = sig.convolve([1,-6,-16],[1,4])
print(num," over ", den)
AG,PG,_ = sig.tf2zpk(num,den)
print("AG = ",AG,"\nPG = ",PG)
#A(s) = (s+4) / (s+1)(s+3)
num = [1,4]
den = [1,4,3]
AA,PA,_ = sig.tf2zpk(num, den)
print(num," over ", den)
print("AA = ",AA,"\nPA = ",PA)
#B(s) = (s+14)(s+12)
num = [1,26,168]
AB = np.roots(num)
print(num)
print("AB = ",AB)
#A(s)G(s) = (s+9) / (s+1)(s+3)(s+2)(s-8)  (two (s+4) terms cancel) from task 3
# UNSTABLE. The s-8 term is in the right half of the cmplx s plane. task 4
num = sig.convolve([1,9],[1,4])
den = sig.convolve([1,4],[1,1])
den = sig.convolve(den,[1,3])
den = sig.convolve(den,[1,2])
den = sig.convolve(den,[1,-8])
print(num," over ", den)
#plotting task 5
tout,ht = sig.step((num,den))
plt.figure(figsize = (10,5))
plt.plot(tout,ht)
plt.grid()
plt.ylabel("Open Loop Step Response Y(s)")
plt.xlabel("s")
plt.title("Task 3.3.5")
#graph goes to infinity around 7. Predicted in task 4. Task 6


#Second half of lab.
numG = [1,9]
denG = sig.convolve([1,-6,-16],[1,4])
numA = [1,4]
denA = [1,4,3]
numB = [1,26,168]
denB = [1]
#task2
#regulation: (gnum*bden) / ( (gden*bden)+(gnum*bnum) )
numGBreg = sig.convolve(numG,denB)
denGBreg = sig.convolve(denG,denB) + sig.convolve(numG,numB)
numH = sig.convolve(numGBreg,numA)
denH = sig.convolve(denGBreg,denA)
AH,PH,_ = sig.tf2zpk(numH,denH)
print(numH," over ", denH)
print("AH = ",AH,"\nPH = ",PH)
# (s+9)(s+4) over (s+5.16+-9.52j)(s+6.18)(s+3)(s+1)
# STABLE. No terms in right half of complex s plane.
#plotting task 4
tout,ht = sig.step((numH,denH))
plt.figure(figsize = (10,5))
plt.plot(tout,ht)
plt.grid()
plt.ylabel("Closed Loop Step Response Y(s)")
plt.xlabel("s")
plt.title("Task 3.4.5")
#The plot tapers off and stabalizes as s increases as expected. Task 5
