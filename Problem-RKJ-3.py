#----------------------------------------------------------------------------------
# Rasmus:
#----------------------------------------------------------------------------------
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from probfit import Chi2Regression, pdf
from scipy import stats
plt.close('all')

# function to create a nice string output
def nice_string_output(names, values, extra_spacing = 2):
    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values):
        string += "{0:s} {1:>{spacing}} \n".format(name, value,
                   spacing = extra_spacing + max_values + max_names - len(name))
    return string[:-2]

# Random numbers
r = np.random
r.seed(42)

SavePlots = False

# Set parameters:
Nnumbers = 10000      # Number of random numbers produced.

# Histogram settings (note the choice of 120 bins):
C = 1./((10*1**(0.1))-(10*0.005**(0.1))) #Calculated from normalization 

Nbins = 100 #0.01 between each bin
xmin = 0.005 #range of the PDF
xmax = 1.005

x_hitmiss = np.zeros(Nnumbers)

#The function we are working with.
def fit_func (x, p3, p4):
    return p3 * p4*x**(-0.9) #P3 would here be the same as C - factor 100 larger

n = 0 #counter
for i in range(Nnumbers):
    #Only using HIT AND MISS! We have a box around the function in the interval - transformation is a Pain in the ass when trying to invert!
    x_hitmiss[i] = 1 #lower boundary on x axis (the lowest value it can have)
    y = C*0.005**(-0.9) #Max function value (max y value - 28.63) 
    while (C*x_hitmiss[i]**(-0.9) < y) :      # ...so keep making new numbers, until this is no longer true!
        x_hitmiss[i] = r.uniform(0.005, 1)
        y = C*0.005**(-0.9) * r.uniform()
        n += 1

print ("Number of tries to do hit and miss to get 10000 numbers:", n)

Hist_HitMiss, Hist_HitMiss_edges = np.histogram(x_hitmiss, bins=Nbins, range=(xmin, xmax))
Hist_HitMiss_centers = 0.5*(Hist_HitMiss_edges[1:] + Hist_HitMiss_edges[:-1])
Hist_HitMiss_error = np.sqrt(Hist_HitMiss)
Hist_HitMiss_indexes = Hist_HitMiss > 0     # Produce a new histogram, using only bins with non-zero entries

#----------------------------------------------------------------------------------
# Plot histograms on screen:
#----------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_title("Random numbers produced by a PDF")
ax.set_xlabel("Random number")
ax.set_ylabel("Frequency / binwidth = 0.01")

chi2_object_hitmiss = Chi2Regression(fit_func, Hist_HitMiss_centers[Hist_HitMiss_indexes], Hist_HitMiss[Hist_HitMiss_indexes], Hist_HitMiss_error[Hist_HitMiss_indexes])
minuit_hitmiss = Minuit(chi2_object_hitmiss, pedantic=False, p3 = 100, p4 = 0.24)
minuit_hitmiss.migrad()

chi2_hitmiss = minuit_hitmiss.fval
#ndof_hitmiss = chi2_object_hitmiss.ndof
ndof_hitmiss  = len(Hist_HitMiss_centers[Hist_HitMiss_indexes]) - len(minuit_hitmiss.args)
prob_hitmiss = stats.chi2.sf(chi2_hitmiss, ndof_hitmiss)

p3, p4 = minuit_hitmiss.args
x_fit = np.linspace(xmin, xmax, 1000)
y_fit_simple = fit_func(x_fit, p3, p4)
ax.plot(x_fit, y_fit_simple, 'r-')

names = ['Hit & Miss:','Entries','Mean','RMS','Chi2 / ndof','Prob','Normalization factor','C']
values = ["",
          "{:d}".format(len(x_hitmiss)),
          "{:.3f}".format(x_hitmiss.mean()),
          "{:.3f}".format(x_hitmiss.std(ddof=1)),
          "{:.3f} / {:d}".format(chi2_hitmiss, ndof_hitmiss),
          "{:.3f}".format(prob_hitmiss),
          "{:.3f} +/- {:.3f}".format(minuit_hitmiss.values['p3'], minuit_hitmiss.errors['p3']),
          "{:.3f} +/- {:.3f}".format(minuit_hitmiss.values['p4'], minuit_hitmiss.errors['p4']),
          ]
ax.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax.transAxes, fontsize=10, verticalalignment='top')

# Drawing histograms with errors in the same plot:
ax.errorbar(Hist_HitMiss_centers[Hist_HitMiss_indexes], Hist_HitMiss[Hist_HitMiss_indexes], Hist_HitMiss_error[Hist_HitMiss_indexes], fmt='r.', capsize=2, capthick=2, label="Hit & Miss")
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.show(block=True)
if (SavePlots) :
    fig.savefig("figures/Assignment-313.pdf", dpi=600)










##################################
#The last part of the question
print ("The last part with getting 1000 numbers which are a collection of 12 random numbers summed together")
Nnumbers = 1000
C = 1./((10*1**(0.1))-(10*0.005**(0.1)))
t = []

n = 0
#Using the same Hit and Miss to produce random numbers
for i in range(Nnumbers):
    x_sum = 0.0
    for j in range(12):
        x_hitmiss = 1 #lower boundary on x axis (the lowest value it can have)
        y = C*0.005**(-0.9) #Max function value (max y value - 28.63) 
        while (C*x_hitmiss**(-0.9) < y) :      # ...so keep making new numbers, until this is no longer true!
            x_hitmiss = r.uniform(0.005, 1)
            y = C*0.005**(-0.9) * r.uniform()
            n += 1
        x_sum +=x_hitmiss
    t.append(x_sum)
print ("Counter for 12 random numbers to fill a list of 1000 entries", n)


t_lampda = np.mean(t)
t_sigma = np.std(t, ddof = 1)
print ("Mean of t list", t_lampda, "with std", t_sigma, "before any fitting.")
       
xmin = 0    #Foudn from min.(A_ill)
xmax = 6   #Same
bins_number = 100 #To get a binwidth of 0.1

counts, bin_edges = np.histogram(t, bins=bins_number, range=(xmin, xmax))
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
s_counts = np.sqrt(counts)

x = bin_centers[counts>0]
y = counts[counts>0]
sy = s_counts[counts>0]

fig1, ax1 = plt.subplots(figsize = (12, 4))
ax1.errorbar(x, y, yerr=sy, label='1000 times sum of 12 numbers',
            fmt='.r',  ecolor='r', elinewidth=1, capsize=1, capthick=1)
ax1.set_title('Assignemnt 3.4 - Sum of 12 random numbers from a PDF a 1000 times')
ax1.set_xlabel('Value of sum of 12 random numbers from the PDF')
ax1.set_ylabel('Frequency / binwidth = 0.06')

##############################
# Draw Gaussian:
# -------------
def func_Gaussian(x, N, mu, sigma) :
    return N * stats.norm.pdf(x, mu, sigma)

Chi2_Gaussian = Chi2Regression(func_Gaussian, x, y, sy)
minuit_Gaussian = Minuit(Chi2_Gaussian, pedantic=False, N=1000, mu=t_lampda, sigma=t_sigma)
minuit_Gaussian.migrad()  # perform the actual fit

xaxis = np.linspace(xmin, xmax, 1000)
yaxis = func_Gaussian(xaxis, *minuit_Gaussian.args)
ax1.plot(xaxis, yaxis, '-', label='Normalised Gaussian distribution')


N_g, mu_g, sigma_g = minuit_Gaussian.args

chi2_val = 0
for x_i, y_i, sy_i in zip(x, y, sy):
    f = func_Gaussian(x_i, N_g, mu_g, sigma_g)
    residual  = ( y_i - f ) / sy_i
    chi2_val += residual**2

DOF = len(x) - len(minuit_Gaussian.args)
chi2_prob =  stats.chi2.sf(chi2_val, DOF)

names = ['Entries', 'Chi2/ndf', 'Prob', 'N', 'mu', 'sigma']
values = ["{:d}".format(len(t)),
          "{:.3f} / {:d}".format(chi2_val, DOF),
          "{:.3f}".format(chi2_prob),
          "{:.3f} +/- {:.3f}".format(minuit_Gaussian.values['N'], minuit_Gaussian.errors['N']),
          "{:.3f} +/- {:.3f}".format(minuit_Gaussian.values['mu'], minuit_Gaussian.errors['mu']),
          "{:.3f} +/- {:.3f}".format(minuit_Gaussian.values['sigma'], minuit_Gaussian.errors['sigma'])
          ]

ax1.text(0.02, 0.95, nice_string_output(names, values), family='monospace', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
plt.legend()
fig1.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig1.savefig("figures/Assignment-314.pdf")












"""
#A little extra: Just to see the function in action with the two starting points - 0 is bad because you may
#not devide by zero: x^(-0.9) = 1/(x^0.9)
print ("\n\n-----------------------------")
print ("MONTE CARLO - 3")
print ("Assigment 3.1:")
print ("\n\nNOT DONE YET\n\n")

#First part of question: 
#Integrer
C_normal_0005 = 1/(((10*1**0.1) - (10*0.005**0.1)))


xaxis = np.linspace(0.005, 1, 1000)
yaxis = C_normal_0005*xaxis**(-0.9) 
 
fig1, ax1 = plt.subplots(figsize=(12, 4))    
ax1.plot(xaxis, yaxis, 'r-', label='norm pdf')
ax1.set_title('Assignemnt 1.3 - Gaussian')
ax1.set_xlabel('Heights')
ax1.set_ylabel('Frequency')
plt.legend()
fig.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig.savefig("Assignment-1.3.pdf")


xaxis = np.linspace(0, 1, 1000)
yaxis = C_normal_0005*xaxis**(-0.9) 

fig2, ax2 = plt.subplots(figsize=(12, 4))    
ax2.plot(xaxis, yaxis, 'r-', label='norm pdf')
ax2.set_title('Assignemnt 1.3 - Gaussian')
ax2.set_xlabel('Heights')
ax2.set_ylabel('Frequency')
plt.legend()
fig.tight_layout()
plt.show(block=False)
if (SavePlots): 
    fig.savefig("Assignment-1.3.pdf")
"""
