#----------------------------------------------------------------------------------
# Rasmus:
#----------------------------------------------------------------------------------

#!/usr/bin/env python
# ----------------------------------------------------------------------------------- #
#
#  ROOT macro for reading data for the Applied Statistics problem set 2017 problem 5.2,
#  regarding timing residuals.
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

data = []
counter = 0

with open( "data_TimingResiduals.txt", 'r' ) as infile :
    for line in infile:
        line = line.strip().split()
        data.append(float(line[0]))
        if (counter < 10) :
            print("  {0:4d}:    {1:6.3f} seconds".format(counter, data[-1]))
        counter += 1

print("  Number of measurements in total: ", counter)



def GetMeanAndErrorOnMean( inlist = [] ) :
# --------------------------------------------------------- #

    if len( inlist ) == 0  :
        print("WARNING: Called function with an empty list")
        return [-9999999999.9, -99999999999.9]

    elif len( inlist ) == 1  :
        print("WARNING: Called function with a list of length one")
        return [inlist[0], -99999999999.9]

    # Find the mean :
    mu = 0.0
    for value in inlist : mu += value
    mu = mu / len( inlist )

    # Find the standard deviation (std) and error on mean (emu):
    std = 0.0
    for value in inlist : std += (value - mu)*(value - mu)
    std = np.sqrt(std / (len(inlist) - 1))
    emu = (std / np.sqrt(len(inlist)))

    return [mu, emu, std]




#---------------------------------------------------------------------------------- 
# Your analysis...
#--------------------------------------------------------------------------
#First dot:
mu, emu, std = GetMeanAndErrorOnMean(data)
print ("\nThe assignment:")
print ("From all residuals: \nMean: {0:.3}, error on mean: {1:.3} and the standard deviation (RMS): {2:.3}".format(mu, emu, std))
print ("The typical timing uncertaintty on one single measurement is hence the std:", std)
print ("and mean of the residuals: {:.3} is aproximately consistent with being 0.".format(mu))



######################################################
#Second dot:
print ("\nSecond dot: See plot")

xaxis = np.array(range(1, counter+1))

#And data is a list with 1726 residuals
sigma_init = 0.0
error_xaxis = 0.0*np.ones(len(xaxis))
error_data = sigma_init * np.ones(len(data))
xmin_ax, xmax_ax = xaxis.min()-0.5, xaxis.max()+0.5

fig, ax = plt.subplots(figsize=(12, 4))
ax.errorbar(xaxis, data, xerr=error_xaxis, yerr=error_data, fmt='.k', ecolor='b', elinewidth=1, capsize=1)
ax.set_xlim(xmin_ax, xmax_ax)
ax.plot([xmin_ax, xmax_ax], [0, 0], '-r', linewidth = 2) #How to plot a read line in the middle (0)
ax.set_title('Assignemnt 5.2 - Residuals')
ax.set_xlabel("Measurements")
ax.set_ylabel("Time residuals (s)")
fig.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig.savefig("figures/Assignment-521.pdf")

#To see outlines: 
outlines =  0.26 #0.32 #0.26 is 3 sigma
strange_residuals = []
for i in range(len(data)):
    if data[i] > outlines or data[i] < - outlines:
        strange_residuals.append(i)

data_strange = []
for i in strange_residuals:
    data_strange.append(data[i])

print ("The {:d} elements that are either more or less than {:.2}:".format(len(strange_residuals), outlines))
print (strange_residuals)
print ("With the values:")
print (data_strange)





######################################################
#Third dot:
#Remember, i have mu and std from the first dot
data_array = np.asarray(data)

xmin = -0.7    #Foudn from min.(A_ill)
xmax = 0.6   #Same
bins_number = 130 #(before 100) #To get a binwidth of 0.01

residual, bin_edges = np.histogram(data_array, bins=bins_number, range=(xmin, xmax))
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
s_residual = np.sqrt(residual)

x = bin_centers[residual>0]
y = residual[residual>0]
sy = s_residual[residual>0]

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.errorbar(x, y, yerr=sy, label='Histogram of residuals',
            fmt='.k',  ecolor='k', elinewidth=1, capsize=2, capthick=1)
ax1.set_title('Assignemnt 5.2 - Histogram of residuals with a Gaussian fit')
ax1.set_xlabel('Residual')
ax1.set_ylabel('Frequency / binwidth = 0.01')


# Draw Gaussian:
# -------------
def func_Gaussian(x, N, mu, sigma) :
    return N * stats.norm.pdf(x, mu, sigma)

Chi2_Gaussian = Chi2Regression(func_Gaussian, x, y, sy)
minuit_Gaussian = Minuit(Chi2_Gaussian, pedantic=False, N=len(y), mu=mu, sigma=std)  
minuit_Gaussian.migrad()  # perform the actual fit

chi2_gaussian = minuit_Gaussian.fval
ndof_gaussian  = len(x) - len(minuit_Gaussian.args)
prob_gaussian = stats.chi2.sf(chi2_gaussian, ndof_gaussian)

xaxis = np.linspace(xmin, xmax, 1000)
yaxis = func_Gaussian(xaxis, *minuit_Gaussian.args)
ax1.plot(xaxis, yaxis, '-', label='Gaussian distribution fit')

names = ['Entries', 'Mean', 'Std Dev', 'Chi2/ndf', 'Prob', 'N fit', 'mu fit', 'sigma fit']
values = ["{:d}".format(len(data_array)),
          "{:.3f}".format(mu),
          "{:.3f}".format(std),
          "{:.3f} / {:d}".format(chi2_gaussian, ndof_gaussian),  
          "{:.3f}".format(prob_gaussian),
          "{:.1f} +/- {:.1f}".format(minuit_Gaussian.values['N'], minuit_Gaussian.errors['N']),
          "{:.3f} +/- {:.3f}".format(minuit_Gaussian.values['mu'], minuit_Gaussian.errors['mu']),
          "{:.3f} +/- {:.3f}".format(minuit_Gaussian.values['sigma'], minuit_Gaussian.errors['sigma']),
          ]

ax1.text(0.02, 0.95, nice_string_output(names, values), family='monospace', transform=ax1.transAxes, fontsize=12, verticalalignment='top')

plt.legend()
fig1.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig1.savefig("figures/Assignment-523.pdf")

print ("Gaussian fit is no go")






#########################################################
#Trying two gaussians: 
#Fourth dot: Remember, i have mu and std from the first dot
print ("Trying to do a double fit instead:")
data_array = np.asarray(data)

xmin = -0.7    #Foudn from min.(A_ill)
xmax = 0.6   #Same
bins_number = 130 #(before 100) #To get a binwidth of 0.01

residual, bin_edges = np.histogram(data_array, bins=bins_number, range=(xmin, xmax))
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
s_residual = np.sqrt(residual)

x = bin_centers[residual>0]
y = residual[residual>0]
sy = s_residual[residual>0]

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.errorbar(x, y, yerr=sy, label='Histogram of residuals',
            fmt='.k',  ecolor='k', elinewidth=1, capsize=2, capthick=1)
ax3.set_title('Assignemnt 5.2 - Histogram of residuals with a double Gaussian fit')
ax3.set_xlabel('Residual')
ax3.set_ylabel('Frequency / binwidth = 0.01')


# Draw Gaussian:
# -------------
def func_double_Gaussian(x, N, mu, sigma, N1, mu1, sigma1) :
    return N * stats.norm.pdf(x, mu, sigma) + N1 * stats.norm.pdf(x, mu1, sigma1)

Chi2_Gaussian = Chi2Regression(func_double_Gaussian, x, y, sy)
minuit_Gaussian = Minuit(Chi2_Gaussian, pedantic=False, N=1, mu=0.001, sigma=0.06, N1=10, mu1=-0.01, sigma1 = 0.15)  
minuit_Gaussian.migrad()  # perform the actual fit

chi2_gaussian = minuit_Gaussian.fval
ndof_gaussian  = len(x) - len(minuit_Gaussian.args)
prob_gaussian = stats.chi2.sf(chi2_gaussian, ndof_gaussian)

xaxis = np.linspace(xmin, xmax, 1000)
yaxis = func_double_Gaussian(xaxis, *minuit_Gaussian.args)
ax3.plot(xaxis, yaxis, '-', label='Double Gaussian fit made of the other Gaussians')

names = ['Entries', 'Chi2/ndf', 'Prob', 'N fit', 'mu fit', 'sigma fit', 'N1 fit', 'mu1 fit', 'sigma1 fit']
values = ["{:d}".format(len(data_array)),
          "{:.3f} / {:d}".format(chi2_gaussian, ndof_gaussian),  
          "{:.3f}".format(prob_gaussian),
          "{:.1f} +/- {:.1f}".format(minuit_Gaussian.values['N'], minuit_Gaussian.errors['N']),
          "{:.3f} +/- {:.3f}".format(minuit_Gaussian.values['mu'], minuit_Gaussian.errors['mu']),
          "{:.3f} +/- {:.3f}".format(minuit_Gaussian.values['sigma'], minuit_Gaussian.errors['sigma']),
          "{:.1f} +/- {:.1f}".format(minuit_Gaussian.values['N1'], minuit_Gaussian.errors['N1']),
          "{:.3f} +/- {:.3f}".format(minuit_Gaussian.values['mu1'], minuit_Gaussian.errors['mu1']),
          "{:.3f} +/- {:.3f}".format(minuit_Gaussian.values['sigma1'], minuit_Gaussian.errors['sigma1']),
          ]

ax3.text(0.02, 0.95, nice_string_output(names, values), family='monospace', transform=ax3.transAxes, fontsize=12, verticalalignment='top')


#And plotting the two Gauss seperately just to see how they look
xaxis = np.linspace(xmin, xmax, 1000)
yaxis = func_Gaussian(xaxis, minuit_Gaussian.values['N'], minuit_Gaussian.values['mu'], minuit_Gaussian.values['sigma'])
ax3.plot(xaxis, yaxis, 'r-', label='Gaussian distribution fit (N fit, mu fit, sigma fit)')

xaxis = np.linspace(xmin, xmax, 1000)
yaxis = func_Gaussian(xaxis, minuit_Gaussian.values['N1'], minuit_Gaussian.values['mu1'], minuit_Gaussian.values['sigma1'])
ax3.plot(xaxis, yaxis, 'c-', label='Gaussian distribution fit (N1 fit, mu1 fit, sigma1 fit)')

plt.legend()
fig3.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig3.savefig("figures/Assignment-524A.pdf")




























######################
#Try something: I am removing the 8 points that i have already found suspisious and make a nother fit
print ("Trying to remove unlikely points")
print ("length before doing anything:", len(data_array))
data_array_before_anything = data_array.copy()

maxgenerations = 100
threshold = 0.0001
remove = 0
t = 0
verbose = True
data_array1 = data_array_before_anything

while (remove < maxgenerations):
    data_array_1less = []
    t += 1
    fucking_prob = [stats.norm.sf(np.abs(xtest - mu), loc = 0, scale = std) for xtest in data_array1]
    print (fucking_prob)
    print (fucking_prob[np.argmin(fucking_prob)])
    if fucking_prob[np.argmin(fucking_prob)] < threshold: 
        if t % 20 == 0:
            print (t, "iterations")
        if (verbose):
            print ("There is a datapoint with low probability: {0:.2E} with index: {1:.0f} - going to be removed"
                   .format(fucking_prob[np.argmin(fucking_prob)], np.argmin(fucking_prob)))
        remove += 1
     
        for i in range(len(data_array1)):
            if i != np.argmin(fucking_prob):
                data_array_1less.append(data_array1[i])
        
        data_array1 = data_array_1less   
        
        mu, emu, std = GetMeanAndErrorOnMean(data_array1)
        if (verbose):
            print("New calculated mean = {:6.4f} +- {:6.4f} m      RMS = {:6.4f} m   (N = {:3d})\n".format(
                    mu, std, emu, len(data_array1)))
    
    else:
        print ("\n\tAll datapoints are now above the threshold: {0:f} with the worst datapoint having a probability of: {1:.2E}".format(threshold, fucking_prob[np.argmin(fucking_prob)]))
        print("Values are: mean = {:6.4f} +- {:6.4f} m      RMS = {:6.4f} m   (N = {:3d})".format(
                    mu, std, emu, len(data_array1)))
        print ("And number of data removed is:", remove)
        break

print ("Length after removing unlikely points:", len(data_array1))


data_array1 = np.asarray(data_array1)

xmin = -0.3    #Foudn from min.(A_ill)
xmax = 0.32  #Same
bins_number = 50 #(before 100) #To get a binwidth of 0.1

residual, bin_edges = np.histogram(data_array1, bins=bins_number, range=(xmin, xmax))
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
s_residual = np.sqrt(residual)

x = bin_centers[residual>0]
y = residual[residual>0]
sy = s_residual[residual>0]

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.errorbar(x, y, yerr=sy, label='Histogram of residuals',
            fmt='.k',  ecolor='k', elinewidth=1, capsize=2, capthick=1)
ax2.set_title('Assignemnt 5.2 - Histogram of residuals with a Cauchy fit')
ax2.set_xlabel('Residual')
ax2.set_ylabel('Frequency / binwidth = 0.0124')


# Draw Gaussian:
# -------------
def func_cauchy(x, N, mu, sigma) :
    return N * stats.cauchy.pdf(x, mu, sigma)

Chi2_cauchy = Chi2Regression(func_cauchy, x, y, sy)
minuit_cauchy = Minuit(Chi2_cauchy, pedantic=False, N=40, mu=mu, sigma=std-0.012)  
minuit_cauchy.migrad()  # perform the actual fit

chi2_cauchy = minuit_cauchy.fval
ndof_cauchy  = len(x) - len(minuit_cauchy.args)
prob_cauchy = stats.chi2.sf(chi2_cauchy, ndof_cauchy)

xaxis = np.linspace(xmin, xmax, 1000)
yaxis = func_cauchy(xaxis, *minuit_cauchy.args)
ax2.plot(xaxis, yaxis, '-', label='Cauchy distribution fit')

names = ['Entries', 'Mean', 'Std Dev', 'Chi2/ndf', 'Prob', 'N fit', 'mu fit', 'sigma fit']
values = ["{:d}".format(len(data_array)),
          "{:.3f}".format(mu),
          "{:.3f}".format(std),
          "{:.3f} / {:d}".format(chi2_cauchy, ndof_cauchy),  
          "{:.3f}".format(prob_cauchy),
          "{:.1f} +/- {:.1f}".format(minuit_cauchy.values['N'], minuit_cauchy.errors['N']),
          "{:.3f} +/- {:.3f}".format(minuit_cauchy.values['mu'], minuit_cauchy.errors['mu']),
          "{:.3f} +/- {:.3f}".format(minuit_cauchy.values['sigma'], minuit_cauchy.errors['sigma']),
          ]

ax2.text(0.02, 0.95, nice_string_output(names, values), family='monospace', transform=ax2.transAxes, fontsize=12, verticalalignment='top')

plt.legend()
fig2.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig2.savefig("figures/Assignment-524B.pdf")





try:
    __IPYTHON__
except:
    raw_input('Press Enter to exit')