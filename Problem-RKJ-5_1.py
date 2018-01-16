#----------------------------------------------------------------------------------
# Rasmus:
#----------------------------------------------------------------------------------

#!/usr/bin/env python
# ----------------------------------------------------------------------------------- #
#
#  ROOT macro for reading data for the Applied Statistics problem set 2017 problem 5.1,
#  regarding Luke Lightning's Lights.
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
from probfit import Chi2Regression , BinnedChi2, BinnedLH, UnbinnedLH
plt.close('all')

SavePlots = False

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

# Define input data lists:
x   = []
y   = []
ex  = [] #equal to zero for all
ey  = [] #equal to 0.11 for all

with open( 'data_LukeLightningLights.txt', 'r' ) as infile :
    for line in infile:
        line = line.strip().split()
        x.append(float(line[0]))
        ex.append(float(line[1]))
        y.append(float(line[2]))
        ey.append(float(line[3]))

        # Print the numbers as a sanity check (only 41 in total):
        print("  Read data:  {0:6.3f}  {1:6.3f}  {2:6.3f}  {3:6.3f}".format(x[-1], ex[-1], y[-1], ey[-1]))





#---------------------------------------------------------------------------------- 
# Your analysis...
#--------------------------------------------------------------------------
#Does the first 12 month have a constant income?
x_12 = np.copy(x[:12])
y_12 = np.copy(y[:12])
ey_12 = np.copy(ey[:12])

fig, ax = plt.subplots(figsize=(12, 4))
ax.errorbar(x_12, y_12, yerr=ey_12, label='Income for 12 months',
            fmt='.r',  ecolor='r', elinewidth=1, capsize=1, capthick=1)
ax.set_title('Assignemnt 5.1 - Monthly income for 12 months')
ax.set_xlabel('Months')
ax.set_ylabel('Income in million dollars')

def fit_function(x, alpha0, alpha1): #So alpha0 + alpha1*x = y
        return alpha0 + alpha1*x

chi2_object = Chi2Regression(fit_function, x_12, y_12, ey_12) 
minuit = Minuit(chi2_object, pedantic=False, alpha0=-0.3, fix_alpha1=True, print_level=0)  
minuit.migrad();  # perform the actual fit

Chi2_fit = minuit.fval                          #The chi2 value
#Ndof_calc = chi2_object.ndof                   #The degree of freedom
Ndof_calc  = len(x_12) - len(minuit.args)
Prob_fit =  stats.chi2.sf(Chi2_fit, Ndof_calc)  # The chi2 probability given N degrees of freedom (Ndof)

xaxis = np.linspace(1, 12, 1000)
ax.plot(xaxis, fit_function(xaxis, *minuit.args), '-r') 

names = ['Chi2/ndf', 'Prob', 'y-intersept', 'slope']
values = ["{:.3f} / {:d}".format(Chi2_fit, Ndof_calc), 
          "{:.3f}".format(Prob_fit), 
          "{:.3f} +/- {:.3f}".format(minuit.values['alpha0'], minuit.errors['alpha0']),
          "{:.3f} +/- {:.3f}".format(minuit.values['alpha1'], minuit.errors['alpha1']),
          ]

ax.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax.transAxes, fontsize=10, verticalalignment='top')

fig.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig.savefig("figures/Assignment-511line.pdf")





################
#TEst if a constant! - 1 parameter
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.errorbar(x_12, y_12, yerr=ey_12, label='Income for 12 months',
            fmt='.r',  ecolor='r', elinewidth=1, capsize=1, capthick=1)
ax1.set_title('Assignemnt 5.1 - Monthly income for 12 months - only a constant')
ax1.set_xlabel('Months')
ax1.set_ylabel('Income in million dollars')


def fit_function(x, alpha0): #So alpha0 + 0*x = y
        return alpha0 + 0*x

chi2_object = Chi2Regression(fit_function, x_12, y_12, ey_12) 
minuit = Minuit(chi2_object, pedantic=False, alpha0=-0.3)  
minuit.migrad();  # perform the actual fit

Chi2_fit = minuit.fval          #The chi2 value
#Ndof_calc = chi2_object.ndof    #The degree of freedom
Ndof_calc  = len(x_12) - len(minuit.args)
Prob_fit =  stats.chi2.sf(Chi2_fit, Ndof_calc) # The chi2 probability given N degrees of freedom (Ndof)

xaxis = np.linspace(1, 12, 1000)
ax1.plot(xaxis, fit_function(xaxis, *minuit.args), '-r') 

names = ['Chi2/ndf', 'Prob', 'y-intersept']
values = ["{:.3f} / {:d}".format(Chi2_fit, Ndof_calc), 
          "{:.3f}".format(Prob_fit), 
          "{:.3f} +/- {:.3f}".format(minuit.values['alpha0'], minuit.errors['alpha0']),
          ]

ax1.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax1.transAxes, fontsize=10, verticalalignment='top')

fig1.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig1.savefig("figures/Assignment-511constant.pdf")










############################################################################
#How long is there a linear relation?
n = 15 # 15 is good: 12 = 6.8%, 13 = 3.9%, 14 = 5.7%, 15 = 1.6%, 16 = 0.3 %, 17 = 0%
x_more = np.copy(x[:n])
y_more = np.copy(y[:n])
ey_more = np.copy(ey[:n])

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.errorbar(x_more, y_more, yerr=ey_more, label='Income for 15 months',
            fmt='.r',  ecolor='r', elinewidth=1, capsize=1, capthick=1)
ax2.set_title('Assignemnt 5.1 - Monthly income for 15 months')
ax2.set_xlabel('Months')
ax2.set_ylabel('Income in million dollars')

def fit_function(x, alpha0, alpha1): #alpha0 + alpha1*x = y
        return alpha0 + alpha1*x

chi2_object = Chi2Regression(fit_function, x_more, y_more, ey_more) 
minuit = Minuit(chi2_object, pedantic=False, alpha0=-0.5, alpha1=0.03, print_level=0)  
minuit.migrad();  # perform the actual fit

Chi2_fit = minuit.fval          #The chi2 value
#Ndof_calc1 = chi2_object.ndof    #The degree of freedom
Ndof_calc  = len(x_more) - len(minuit.args)
Prob_fit =  stats.chi2.sf(Chi2_fit, Ndof_calc) # The chi2 probability given N degrees of freedom (Ndof)

xaxis = np.linspace(1, n, 1000)
ax2.plot(xaxis, fit_function(xaxis, *minuit.args), '-r') 

names = ['Chi2/ndf', 'Prob', 'y-intersept', 'slope']
values = ["{:.3f} / {:d}".format(Chi2_fit, Ndof_calc), 
          "{:.3f}".format(Prob_fit), 
          "{:.3f} +/- {:.3f}".format(minuit.values['alpha0'], minuit.errors['alpha0']),
          "{:.3f} +/- {:.3f}".format(minuit.values['alpha1'], minuit.errors['alpha1']),
          ]

ax2.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax2.transAxes, fontsize=10, verticalalignment='top')

fig2.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig2.savefig("figures/Assignment-512.pdf")








#############################################################################
#between 26 and 31
n1, n2 = 25, 31 
x_around31 = np.copy(x[n1:n2])
y_around31 = np.copy(y[n1:n2])
ey_around31 = np.copy(ey[n1:n2])

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.errorbar(x_around31, y_around31, yerr=ey_around31, label='Income for month 26-31',
            fmt='.r',  ecolor='r', elinewidth=1, capsize=1, capthick=1)
ax3.set_title('Assignemnt 5.1 - Monthly income for month 26-31')
ax3.set_xlabel('Months')
ax3.set_ylabel('Income in million dollars')


def fit_function(x, alpha0, alpha1): #alpha0 + alpha1*x = y
        return alpha0 + alpha1*x

chi2_object = Chi2Regression(fit_function, x_around31, y_around31, ey_around31) 
minuit = Minuit(chi2_object, pedantic=False, alpha0=2, alpha1=0.059, print_level=0)  
minuit.migrad();  # perform the actual fit

Chi2_fit = minuit.fval          #The chi2 value
#Ndof_calc = chi2_object.ndof    #The degree of freedom
Ndof_calc  = len(x_around31) - len(minuit.args)
Prob_fit =  stats.chi2.sf(Chi2_fit, Ndof_calc) # The chi2 probability given N degrees of freedom (Ndof)

xaxis = np.linspace(n1+1, n2, 1000)
ax3.plot(xaxis, fit_function(xaxis, *minuit.args), '-r') 

names = ['Chi2/ndf', 'Prob', 'y-intersept', 'slope']
values = ["{:.3f} / {:d}".format(Chi2_fit, Ndof_calc), 
          "{:.3f}".format(Prob_fit), 
          "{:.3f} +/- {:.3f}".format(minuit.values['alpha0'], minuit.errors['alpha0']),
          "{:.3f} +/- {:.3f}".format(minuit.values['alpha1'], minuit.errors['alpha1']),
          ]

ax3.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax3.transAxes, fontsize=10, verticalalignment='top')

fig3.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig3.savefig("figures/Assignment-513.pdf")

alpha0_fit, alpha1_fit = minuit.values['alpha0'], minuit.values['alpha1']
sigma_alpha0_fit, sigma_alpha1_fit = minuit.errors['alpha0'], minuit.errors['alpha1']

#The equation: y = p1*x+p0:
n_32_estimate = alpha1_fit * 32 + alpha0_fit
n_32_sigma = np.sqrt(sigma_alpha0_fit**2 + sigma_alpha1_fit**2)

sigma_difference = np.sqrt(n_32_sigma**2 + 0.110**2)

print ("Estiamted value for n=32 if no disruptive cost change: {0:.3} and its uncertainty {1:.3}".format(n_32_estimate, n_32_sigma))
print ("So there is a change of: {:.3} +/- {:.3}".format(n_32_estimate-y[31], sigma_difference))












#############################################################################
#The entire range
fig4, ax4 = plt.subplots(figsize=(12, 4))
ax4.errorbar(x, y, yerr=ey, label='Income for all month',
            fmt='.r',  ecolor='r', elinewidth=1, capsize=1, capthick=1)
ax4.set_title('Assignemnt 5.1 - Monthly income for month 26-31')
ax4.set_xlabel('Months')
ax4.set_ylabel('Income in million dollars')


def func_advanced(x, p0, p1, p2, p3, p4, p5):
    if x < 32:
        return (p0 * 1./(1+np.exp(-(x-p2)*p3)) + p1)
    else:
        return p4 + p5*x
func_advanced_vec = np.vectorize(func_advanced)
# Above is a function that transforms a non-vectorized function to vectorized,
# such that func_advanced_vec allows x to be a numpy array instead of just a scalar
x = np.asarray(x)
y = np.asarray(y)
ey = np.asarray(ey)

chi2_object_advanced = Chi2Regression(func_advanced, x, y, ey)
minuit_advanced = Minuit(chi2_object_advanced, pedantic=False, 
                         p0 = 4, p1 = 1, p2=20, p3=5.0, 
                         p4=0.5, p5=1)
minuit_advanced.migrad() # fit

p0, p1, p2, p3, p4, p5 = minuit_advanced.args
print("Over the entire data range: \nThe advanced fit gives the following parameters:")
for name in minuit_advanced.parameters:
    print("Fit value: {0} = {1:.5f} +/- {2:.5f}".format(name, minuit_advanced.values[name], minuit_advanced.errors[name]))

x_fit = np.linspace(1, 41, 1000)
y_fit_advanced = func_advanced_vec(x_fit, *minuit_advanced.args)
ax4.plot(x_fit, y_fit_advanced, '-')

chi2_val = 0                                            #The Chi2 value
for x_i, y_i, sy_i in zip(x, y, ey):
    f = func_advanced(x_i, p0, p1, p2, p3, p4, p5)
    residual  = ( y_i - f ) / sy_i
    chi2_val += residual**2

DOF = len(x) - len(minuit_advanced.args)
chi2_prob =  stats.chi2.sf(chi2_val, DOF)

names = ['Entries', 'Chi2/ndf', 'Prob','p0','p1', 'p2', 'p3', 'The linear part:', 'y-intersept', 'slope']
values = ["{:d}".format(len(x)),
          "{:.3f} / {:d}".format(chi2_val, DOF),
          "{:.3f}".format(chi2_prob),
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p0'], minuit_advanced.errors['p0']),
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p1'], minuit_advanced.errors['p1']),          
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p2'], minuit_advanced.errors['p2']),
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p3'], minuit_advanced.errors['p3']),
          "",
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p4'], minuit_advanced.errors['p4']),
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p5'], minuit_advanced.errors['p5'])
          ]

# place a text box in upper left in axes coords
ax4.text(0.02, 0.95, nice_string_output(names, values), family='monospace', transform=ax4.transAxes, fontsize=12, verticalalignment='top')


fig4.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig4.savefig("figures/Assignment-514.pdf")








"""
#With 4 linear:
fig4, ax4 = plt.subplots(figsize=(12, 4))
ax4.errorbar(x, y, yerr=ey, label='A for ill people',
            fmt='.r',  ecolor='r', elinewidth=1, capsize=1, capthick=1)
ax4.set_title('Assignemnt 5.3 - Monthsly income')
ax4.set_xlabel('Months')
ax4.set_ylabel('Income in million dollars')


def func_advanced(x, p0, p1, p2, p3, p4, p5, p6, p7) :
    if x < 14:
        return p0 + p1*np.sin(x)
    elif x < 26:
        return p2 + p3*x
    elif x < 32: 
        return p4 + p5*x
    else:
        return p6 + p7*x
func_advanced_vec = np.vectorize(func_advanced)
# Above is a function that transforms a non-vectorized function to vectorized,
# such that func_advanced_vec allows x to be a numpy array instead of just a scalar
x = np.asarray(x)
y = np.asarray(y)
ey = np.asarray(ey)

chi2_object_advanced = Chi2Regression(func_advanced, x, y, ey)

minuit_advanced = Minuit(chi2_object_advanced, pedantic=False, 
                         p0=0., p1=2, 
                         p2=0.0, p3=2.0, 
                         p4=0.5, fix_p5=True, 
                         p6=0.6, fix_p7=True)
minuit_advanced.migrad() # fit
p0, p1, p2, p3, p4, p5, p6, p7 = minuit_advanced.args
print("Advanced fit")
for name in minuit_advanced.parameters:
    print("Fit value: {0} = {1:.5f} +/- {2:.5f}".format(name, minuit_advanced.values[name], minuit_advanced.errors[name]))

x_fit = np.linspace(1, 41, 1000)
y_fit_advanced = func_advanced_vec(x_fit, p0, p1, p2, p3, p4, p5, p6, p7)
ax4.plot(x_fit, y_fit_advanced, '-')
# if you have many parameters and dont want to write out all the names,
# you can write it the following way (which is called argument unpacking (the asterix *))
# p = minuit.args # save all the parameters as the tuple p
# y_fit_advanced = func_advanced_vec(x_fit, *p) # input while unpacking the p

chi2_val = 0
for x_i, y_i, sy_i in zip(x, y, ey):
    f = func_advanced(x_i, p0, p1, p2, p3, p4, p5, p6, p7)
    residual  = ( y_i - f ) / sy_i
    chi2_val += residual**2

DOF = len(x) - len(minuit_advanced.args)

chi2_prob =  stats.chi2.sf(chi2_val, DOF)

names = ['Entries', 'Chi2/ndf', 'Prob', 'p0', 'p1', 'p2',
         'p3', 'p4', 'p5', 'p6', 'p7']
values = ["{:d}".format(len(x)),
          "{:.3f} / {:d}".format(chi2_val, DOF),
          "{:.3f}".format(chi2_prob),
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p0'], minuit_advanced.errors['p0']),
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p1'], minuit_advanced.errors['p1']),
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p2'], minuit_advanced.errors['p2']),
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p3'], minuit_advanced.errors['p3']),
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p4'], minuit_advanced.errors['p4']),
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p5'], minuit_advanced.errors['p5']),
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p6'], minuit_advanced.errors['p6']),
          "{:.3f} +/- {:.3f}".format(minuit_advanced.values['p7'], minuit_advanced.errors['p7'])
          ]

# place a text box in upper left in axes coords
ax4.text(0.02, 0.95, nice_string_output(names, values), family='monospace', transform=ax2.transAxes, fontsize=12, verticalalignment='top')


fig4.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig4.savefig("Assignment-5.4.pdf")
"""






try:
    __IPYTHON__
except:
    raw_input('Press Enter to exit')
    