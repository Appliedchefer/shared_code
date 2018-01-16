#----------------------------------------------------------------------------------
# Rasmus:
#----------------------------------------------------------------------------------

#!/usr/bin/env python
# ----------------------------------------------------------------------------------- #
#
#  ROOT macro for reading data for the Applied Statistics problem set 2017 problem 4.1,
#  regarding Fisher's Syndrome.
#
#  Author: Troels C. Petersen (NBI)
#  Email:  petersen@nbi.dk
#  Date:   11th of November 2017
#
# ----------------------------------------------------------------------------------- #

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from iminuit import Minuit
from probfit import Chi2Regression , BinnedChi2 # , BinnedLH#, , UnbinnedLH, , , Extended
plt.close('all')

SavePlots = False


# function to create a nice string output
def nice_string_output(names, values, extra_spacing = 2):
    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values):
        string += "{0:s} {1:>{spacing}} \n".format(name, value,
                   spacing = extra_spacing + max_values + max_names - len(name))
    return string[:-2]

# ----------------------------------------------------------------------------------- #
# Read data:
# ----------------------------------------------------------------------------------- #
A_ill = []
B_ill = []
C_ill = []
A_fresh = []
B_fresh = []
C_fresh = []

data = []

with open( 'data_FisherSyndrome.txt', 'r' ) as infile :
    counter = 0

    for line in infile:
        line = line.strip().split()
        data.append([float(line[2]), float(line[3]), float(line[4]), int(line[0])])

        isIll = data[-1][3]
        A = data[-1][0]
        B = data[-1][1]
        C = data[-1][2]
        
        #My stuff
        if isIll == 1: 
            A_ill.append(A)
            B_ill.append(B)
            C_ill.append(C)
        else: 
            A_fresh.append(A)
            B_fresh.append(B)
            C_fresh.append(C)
        #End of my stuff

        # Print some numbers as a sanity check:
        #if (counter < 10) :
        #    print("  Reading data:   Is the person ill? {0:d}   A = {1:5.2f}   B = {2:5.2f}   C = {3:5.2f}".format(isIll, A, B, C))
        counter += 1
print ("Number of lines read:", counter)



#---------------------------------------------------------------------------------- 
# Your analysis...
#--------------------------------------------------------------------------
###################################################################
#What distributions does A seems to follow for ill people?

"""
#The plot below is purely for easy to see the distribution before a better plot were made
fig1, ax1 = plt.subplots(figsize=(12, 4))
Nbins = 300
ax1.hist(A_ill, bins=Nbins, histtype='step', linewidth=2,label='Gaussian ($\mu$ = 14.05)', color='blue')
ax1.set_title('Assignemnt 1.3 - Gaussian')
ax1.set_xlabel('Heights')
ax1.set_ylabel('Frequency')
plt.legend()
fig1.tight_layout()
plt.show(block=False)
#if (SavePlots): 
#    fig1.savefig("Assignment-1.3.pdf")
"""

#The better plot: 
A_ill_mean = np.mean(A_ill)
xmin = 4    #Foudn from min.(A_ill)
xmax = 24   #Same
bins_number = 200 #To get a binwidth of 0.1

counts, bin_edges = np.histogram(A_ill, bins=bins_number, range=(xmin, xmax))
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
s_counts = np.sqrt(counts)

x = bin_centers[counts>0]
y = counts[counts>0]
sy = s_counts[counts>0]

fig, ax = plt.subplots(figsize = (12, 4))
ax.errorbar(x, y, yerr=sy, label='A for ill people',
            fmt='.r',  ecolor='r', elinewidth=1, capsize=1, capthick=1)
ax.set_xlim(xmin, xmax)
ax.set_title('Assignemnt 4.1 - A for ill people')
ax.set_xlabel('Value of A for ill people')
ax.set_ylabel('Frequency / binwidth = 0.1')

#The fit:
def func_Gaussian(x, N, mu, sigma) :
    return N * stats.norm.pdf(x, mu, sigma)

Chi2_object_Gaussian = Chi2Regression(func_Gaussian, x, y, sy)
minuit_Gaussian = Minuit(Chi2_object_Gaussian, pedantic=False, N=len(A_ill), mu=A_ill_mean, sigma=np.std(A_ill, ddof=1)) #   
minuit_Gaussian.migrad()  # perform the actual fit

xaxis = np.linspace(xmin, xmax, 1000)
yaxis = func_Gaussian(xaxis, *minuit_Gaussian.args)
ax.plot(xaxis, yaxis, '-', label='Gaussian distribution fit')

chi2_Gaussian = minuit_Gaussian.fval
ndof_Gaussian = len(x) - len(minuit_Gaussian.args)
prob_Gaussian = stats.chi2.sf(chi2_Gaussian, ndof_Gaussian)

names = ['Entries', 'Mean', 'Std Dev', 'Chi2/ndf', 'Prob', 'N', 'mu', 'sigma']
values = ["{:d}".format(len(A_ill)),
          "{:.3f}".format(np.mean(A_ill)),
          "{:.3f}".format(np.std(A_ill, ddof=1)),
          "{:.3f} / {:d}".format(chi2_Gaussian, ndof_Gaussian),  
          "{:.3f}".format(prob_Gaussian),
          "{:.1f} +/- {:.1f}".format(minuit_Gaussian.values['N'], minuit_Gaussian.errors['N']),
          "{:.3f} +/- {:.3f}".format(minuit_Gaussian.values['mu'], minuit_Gaussian.errors['mu']),
          "{:.3f} +/- {:.3f}".format(minuit_Gaussian.values['sigma'], minuit_Gaussian.errors['sigma']),
          ]

ax.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax.transAxes, fontsize=10, verticalalignment='top')
plt.legend()
fig.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig.savefig("figures/Assignment-41.pdf")





###################################################################
#What is the linear correlation between variables B and C for ill people?

#First is a plot made and a straight line is fittet to it - for fun
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(B_ill, C_ill, 'o')
ax2.set_title('Assignemnt 4.2 - B vs C for ill people')
ax2.set_xlabel('B ill')
ax2.set_ylabel('C ill')

x = np.asarray(B_ill)
y = np.asarray(C_ill)

#For the fit
ex = 0.45 * np.ones_like(x) #I do not understand this difference!!!
ey = 0.7 * np.ones_like(x)  #I do not understand this difference!!!

def fit_function(x, alpha0, alpha1): #So alpha0 + alpha1*x = y
        return alpha0 + alpha1*x

chi2_object = Chi2Regression(fit_function, x, y, ex, ey) 
minuit = Minuit(chi2_object, pedantic=False, alpha0=1, alpha1=0.03, print_level=0)  
minuit.migrad();  # perform the actual fit

Chi2_fit = minuit.fval # the chi2 value
Ndof_calc = len(x) - len(minuit.args) #Ndof
Prob_fit =  stats.chi2.sf(Chi2_fit, Ndof_calc) #The chi2 probability given N degrees of freedom (Ndof)

xaxis = np.linspace(30, 70, 1000)
ax2.plot(xaxis, fit_function(xaxis, *minuit.args), '-r') 

names = ['Chi2/ndf', 'Prob', 'y-intersept', 'slope']
values = ["{:.3f} / {:d}".format(Chi2_fit, Ndof_calc), 
          "{:.3f}".format(Prob_fit), 
          "{:.3f} +/- {:.3f}".format(minuit.values['alpha0'], minuit.errors['alpha0']),
          "{:.3f} +/- {:.3f}".format(minuit.values['alpha1'], minuit.errors['alpha1']),
          ]
ax2.text(0.01, 0.25, nice_string_output(names, values), family='monospace', transform=ax2.transAxes, fontsize=10, verticalalignment='top')

fig2.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig2.savefig("figures/Assignment-42.pdf")


print ("Different test to find the correlation - probablly more direct since need to start in 0 (ones plot for orrect slope)")
print ("Correlation and probability for no correlation", stats.pearsonr(x,y))

"""
#Old use, probably not needed:
alpha0_fit = minuit.values['alpha0']
alpha1_fit = minuit.values['alpha1']
sigma_alpha0_fit = minuit.errors['alpha0']
sigma_alpha1_fit = minuit.errors['alpha0']
"""





#########################################################################
#Separation between healthy and ill people - use of C (best of the three) and F
print ("\nSeperation of ill and healthy people")
#Making everything into arrays to make everything easier
A_ill_array = np.array(A_ill)
B_ill_array = np.array(B_ill)
C_ill_array = np.array(C_ill)
F_ill = (-1.33*A_ill_array + 0.63* B_ill_array + 2.1* C_ill_array)
#print ("mean C ill", np.mean(C_ill))
#print ("mean F ill", F_ill.mean())

A_fresh_array = np.array(A_fresh)
B_fresh_array = np.array(B_fresh)
C_fresh_array = np.array(C_fresh)
F_fresh = (-1.33*A_fresh_array + 0.63* B_fresh_array + 2.1* C_fresh_array)
#print ("mean c fresh", np.mean(C_fresh))
#print ("mean F fresh", F_fresh.mean())


#Two histograms of C_ill and C_fresh (most seperated of A, B, and C)
fig5, ax5 = plt.subplots(figsize=(12, 4))
Nbins = 140
ax5.hist(C_ill_array, bins=Nbins, histtype='step', linewidth=2,label='Gaussian for C_ill ($\mu$ = -0,29)', color='blue', range = (-4, 3))
ax5.hist(C_fresh_array, bins=Nbins, histtype='step', linewidth=2,label='Gaussian for C_healthy ($\mu$ = 0,55)', color='red', range = (-4, 3))
ax5.set_title('Assignemnt 4.3 - Compare C of ill and healthy people')
ax5.set_xlabel('Value of C in blood')
ax5.set_ylabel('Frequency / binwidth = 0.05')
plt.legend()
fig5.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig5.savefig("figures/Assignment-43C.pdf")


#To get alpha and beta from the histrograms
#We are looking at the ill people (lukkily on the left of healthy and want as few of them said to be healthy when sick - type 1 error)
counts_ill, bin_edges_ill = np.histogram(C_ill_array, bins=200, range=[-4.0, 3.0])
sum_C_ill = np.sum(counts_ill) #The full sum of all counts - whole C

accum_C_ill = 0.0 #The accumulated F
for i in range(len(counts_ill)):
    accum_C_ill += counts_ill[i]
    proc_C_ill = accum_C_ill / sum_C_ill
    if proc_C_ill > 0.99:
        element_alpha = i
        print ("Number of bins in", i)
        break
print ("alfa", proc_C_ill)


counts_fresh, bin_edges_fresh = np.histogram(C_fresh_array, bins=200, range=[-4.0, 3.0])
sum_C_fresh = np.sum(counts_fresh)

accum_C_fresh = 0.0
for i in range(element_alpha):
    accum_C_fresh += counts_fresh[i]
proc_C_fresh = accum_C_fresh / sum_C_fresh
print ("Beta", proc_C_fresh)
print ("So to get a low Type 1 error (sick but said to be healthy - alpha), we get a high type 2 errors (beta")


####
#FOR F now!
####
print ("\nNow for F")

fig5, ax5 = plt.subplots(figsize=(12, 4))
Nbins = 200
ax5.hist(F_ill, bins=Nbins, histtype='step', linewidth=2,label='Gaussian for F_ill ($\mu$ = 12,31)', color='blue', range = (0.0, 30))
ax5.hist(F_fresh, bins=Nbins, histtype='step', linewidth=2,label='Gaussian for F_healthy ($\mu$ = 22,97)', color='red', range = (0.0, 30))
ax5.set_title('Assignemnt 4.3 - Compare F of ill and healthy people')
ax5.set_xlabel('Value of F in blood')
ax5.set_ylabel('Frequency / binwidth = 0.15')
plt.legend()
fig5.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig5.savefig("figures/Assignment-43F.pdf")


#To get alpha and beta from the histrograms
counts_F_ill, bin_edges_ill = np.histogram(F_ill, bins=200, range=[0.0, 30.0])
sum_F_ill = np.sum(counts_F_ill)

accum_F_ill = 0.0 #The accumulated F
for i in range(len(counts_ill)):
    accum_F_ill += counts_F_ill[i]
    proc_F_ill = accum_F_ill / sum_F_ill
    if proc_F_ill > 0.99:
        element_F_alpha = i
        print ("Number of bins in", i)
        break
print ("alfa", proc_F_ill)


counts_F_fresh, bin_edges_fresh = np.histogram(F_fresh, bins=200, range=[0.0, 30.0])
sum_F_fresh = np.sum(counts_F_fresh)

accum_F_fresh = 0.0
for i in range(element_F_alpha):
    accum_F_fresh += counts_F_fresh[i]
proc_F_fresh = accum_F_fresh / sum_F_fresh
print ("Beta", proc_F_fresh)
print ("Because it is a better seperation of healthy and ill people a low alfa now also gives a low beta")




"""
#For a 2D plot for fun
fig5, ax5 = plt.subplots(figsize=(12, 4))
Nbins = 300
ax5.plot(A_ill, F_ill, '.')
ax5.plot(A_fresh, F_fresh, '.')
#ax5.hist(C_fresh, bins=Nbins, histtype='step', linewidth=2,label='Gaussian for C_healthy($\mu$ = ??)', color='red')
ax5.set_title('Assignemnt 1.3 - Gaussian')
ax5.set_xlabel('Heights')
ax5.set_ylabel('Frequency')
plt.legend()
fig5.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig1.savefig("Assignment-4.3.pdf")
"""




try:
    __IPYTHON__
except:
    raw_input('Press Enter to exit')