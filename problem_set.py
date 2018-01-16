from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from probfit import BinnedLH, Chi2Regression, Extended
from scipy.special import erfc
from scipy import stats

r = np.random
r.seed(42)
SavePlots = True
# Functions

def nice_string_output(names, values, extra_spacing = 2):
    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values):
        string += "{0:s} {1:>{spacing}} \n".format(name, value,
                   spacing = extra_spacing + max_values + max_names - len(name))
    return string[:-2]

def mu_RMS (input_data = []):

    N = len(input_data)

    mu = np.sum(input_data)/N

    RMS = np.sqrt(np.sum((input_data-mu)**2)/(N-1))

    errmu = RMS/np.sqrt(N)

    return mu, RMS, errmu

def chi2_my_func (input_data = []):
    N = len(input_data)
    input_data = np.array(input_data)
    top_sum = 0.0
    bottom_sum = 0.0

    for i in range (N):
        top_sum += (input_data [i][0])/(input_data [i][1]**2)
        bottom_sum += 1/(input_data [i][1]**2)



    wmu = top_sum / bottom_sum


    error_mean = np.sqrt(1/bottom_sum)
    chi2 = 0.0
    for i in range(N):
        chi2 += (input_data [i][0]-wmu)**2/(input_data [i][1]**2)


    Ndof = N-1

    prob = stats.chi2.sf(chi2,Ndof)
    if prob > 0.99 or prob < 0.01:
        print ("This probability is {:1.5f}, which is either too large or low a probability to be trusted.".format(prob))
    return wmu, error_mean, chi2, Ndof, prob


#Problem 1 - Probabilities and distributions
#1.1 - Use binominal. Works with example on page 27 from Barlow.
p1 = 1./6.
n1 = 4.0
p2 = 1./36.# probability of event happening in experiment.
n2 = 24.0 # total number of times the experiment is done.
one = stats.binom.pmf(1,n1,p1)
two = stats.binom.pmf(1,n2,p2)

total_2 = 0.0
 #Starting probability (set to 0 so we can add onto it)
for i in range (1, 25): #In which range should the experiment count from? See example on p. 27 in Barlow
# remember that the range function does not take the last number into account!
    total_2 += stats.binom.pmf(i,n2,p2) # For every i in range it calculates a probability and adds to the
    # previous probability.

total_1 = 0.0
for i in range (1, 5):
    print (i)
    total_1 += stats.binom.pmf(i, n1, p1)
print (total_2*100) #multiply by 100 to get it in percents.
print ("two",total_2*100)

print("one", total_1*100)

# 1.2 - Use Poisson.
lambda_mean = 19.3

test_uden_ida = stats.poisson.pmf(42, lambda_mean)
test = (1-stats.poisson.pmf(42, lambda_mean))**1930



prob_signif = 1-test
print (test_uden_ida*1000000000000)
print("test",test)
print ("prob_sig", prob_signif)

#Alternatively use 1- (1-stats.poisson.sf(42-1, lambda_mean))**1930
bb = 1- (stats.poisson.cdf(42-1, lambda_mean))**1930
print("The poisson prob is: {:}".format(bb))

#Not sure what is ment with the question.


# 1.3 - Use Gaussian

gauss_mean = 1.69
gauss_rms = 0.06

test_gauss = 1-stats.norm.cdf(1.85,gauss_mean, gauss_rms) #could also take the sf function. cdf takes from 0 to the chosen point.

print ("gauss new",test_gauss*100)



times = 10000

gauss = np.random.normal(gauss_mean, gauss_rms, times)

print (gauss)


#get the 80%

top_80 = np.percentile(gauss, 80)

print ("her top80", top_80)

#append all the hights above 80% to a list and then calculate the average.
new_list = []
for i in range (len(gauss)):
    if gauss[i] > top_80:
        new_list.append(gauss[i])

average = sum(new_list)/len(new_list)
print ("average hights",sum(new_list)/len(new_list))
rms_average = gauss_rms/np.sqrt(times)
print (rms_average)

gauss_mu, gauss_rms, gauss_errmu = mu_RMS(new_list)
print (gauss_errmu, )

#Problem 2 - Error propagation
#2.1
#This error propagation on error propagation in the sense that you need to take into
# account that r is squared. should be in relation L 1:2 r
#2.2
#variables
m = 0.0084
sigma_m = 0.0005
velocity = np.array([3.61, 2.00, 3.90, 2.23, 2.32, 2.48, 2.43, 3.86, 4.43, 3.78])

#2.2.1 Is the uncertanty the error on the mean or the width?
mean_velocity, sigma_velocity, err_mean_vel = mu_RMS(velocity)

err_mean_vel = sigma_velocity/np.sqrt(10)
print ("mean_vel, and error on mean",mean_velocity, err_mean_vel)

#2.2.2

E_kin = 0.5 * m * mean_velocity**2
dEdm = 0.5*mean_velocity**2
dEdv = m* mean_velocity


sigma_E = np.sqrt(dEdm**2*sigma_m**2+dEdv**2*err_mean_vel**2)

print ("The kinetic energy and its uncertanty is: {0:3.4f} +- {1: 3.4f}".format(E_kin, sigma_E))

#2.2.3
sigma_E_m_only = np.sqrt(dEdm**2*sigma_m**2)
sigma_E_velocity_only = np.sqrt(dEdv**2*err_mean_vel**2)

print ("Impact on sigma E from m is: {0:3.4f}. The impact on sigma E from v is {1:3.4f}".format(sigma_E_m_only, sigma_E_velocity_only))

#HOW TO FIND N?!?!?! The error on E from m is almost 17 times samller that the value of E
#The error on E from v is around 5.5 times smaller that the value of E
#The ratio bestween the two impacts are basically 3, hence the number of measurements
# to make the in order to make them impact equally should be 3 times the current measurements,
# but since v is squared in the formula, so should it be when making measurements hence 9 times, 90 i total.


# Problem 3 - Monte Carlo f(x) = x^-0.9, [0.005:1]. 1^-0.9 = 1, 0.005^-0.9 = 117.74

#3.1 Find C through normilazation. C is around 0.25. see notes.

#3.2
def fit_function(x,p0):
    return p0*x**(-0.9)
Npoints = 10000 #number of times to run the experiment
Nbins = 120
xmin =0.005
xmax = 1
C=0.2431338987
x_hitmiss = np.zeros(Npoints)

n = 0.0
for i in range (Npoints):

    x_hitmiss[i] = 1.0
    y = C*0.005**(-0.9)
    while C*x_hitmiss[i]**(-0.9) < y:
        x_hitmiss[i] = r.uniform(0.005, 1)
        y  = C*0.005**(-0.9)* r.uniform()
        n += 1
print (len(x_hitmiss), n)

Hist_HitMiss, Hist_HitMiss_edges = np.histogram(x_hitmiss, bins=Nbins, range=(xmin, xmax))
Hist_HitMiss_centers = 0.5*(Hist_HitMiss_edges[1:] + Hist_HitMiss_edges[:-1])
Hist_HitMiss_error = np.sqrt(Hist_HitMiss)
Hist_HitMiss_indexes = Hist_HitMiss > 0     # Produce a new histogram, using only bins with non-zero entries


#----------------------------------------------------------------------------------
# Plot histograms on screen:
#----------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Random number")
ax.set_ylabel("Frequency")

chi2_object_hitmiss = Chi2Regression(fit_function, Hist_HitMiss_centers[Hist_HitMiss_indexes], Hist_HitMiss[Hist_HitMiss_indexes], Hist_HitMiss_error[Hist_HitMiss_indexes])
minuit_hitmiss = Minuit(chi2_object_hitmiss,p0=20.0,  pedantic=False)

minuit_hitmiss.migrad()
chi2_hitmiss = minuit_hitmiss.fval
ndof_hitmiss = chi2_object_hitmiss.ndof
prob_hitmiss = stats.chi2.sf(chi2_hitmiss, ndof_hitmiss)


p0 = minuit_hitmiss.args
x_fit = np.linspace(xmin, xmax, 1000)

y_fit_simple = fit_function(x_fit,p0)
#ax.plot(x_fit, y_fit_simple, 'b-')

names = ['Hit & Miss:','Entries','Mean','RMS']
values = ["",
          "{:d}".format(len(x_hitmiss)),
          "{:.3f}".format(x_hitmiss.mean()),
          "{:.3f}".format(x_hitmiss.std(ddof=1)),
          ]
ax.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax.transAxes, fontsize=10, verticalalignment='top')

# Drawing histograms with errors in the same plot:

ax.errorbar(Hist_HitMiss_centers[Hist_HitMiss_indexes], Hist_HitMiss[Hist_HitMiss_indexes], Hist_HitMiss_error[Hist_HitMiss_indexes], fmt='r.', capsize=2, capthick=2, label="Hit & Miss")
ax.set_ylim(bottom=0)

if SavePlots:
    fig.savefig("hitandmiss.png")


#3.4

t = []


for i in range (1000):
    x_sum = 0.0
    for i in range(12):
        x_hitmiss = 1.0
        y = C*0.005**(-0.9)
        while C*x_hitmiss**(-0.9) < y:
            x_hitmiss = r.uniform(0.005, 1)
            y  = C*0.005**(-0.9)* r.uniform()
        x_sum += x_hitmiss
    t.append(x_sum)


#print (mean_t, sigma_t)
t = np.array(t)
t_max = max(t)
fig1, ax1 = plt.subplots(figsize = (10,5))
hist_monte = ax1.hist(t, bins=100, range=(0, t_max+1), histtype='step', linewidth=2, label='Gaussian ($\mu$ = -0.5)', color='blue')
ax1.set_xlabel("Sum of t")
ax1.set_ylabel("Frequency")



counts_ill, bin_edges_ill = np.histogram(t, bins=100, range=(0, 7.0))
bin_centers_ill = (bin_edges_ill[1:] + bin_edges_ill[:-1])/2
s_counts_ill = np.sqrt(counts_ill)

x = bin_centers_ill[counts_ill>0]

y = counts_ill[counts_ill>0]

sy = s_counts_ill[counts_ill>0]

fig2, ax2 = plt.subplots (figsize = (10,5))
ax2.errorbar (x, y, yerr=sy, label='Sum of t',
            fmt='.k',  ecolor='k', elinewidth=1, capsize=1, capthick=1)

A_ill_mean, A_ill_sigma, A_ill_error_mean = mu_RMS(t)

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return stats.norm.pdf(x,mu,sigma)#1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def gauss_extended(x, N, mu, sigma) :
    """Non-normalized Gaussian"""
    return N * gauss_pdf(x, mu, sigma)




chi_A_ill = Chi2Regression(gauss_extended, x, y, sy)
minuit_ill = Minuit(chi_A_ill, pedantic=False, N=len(t), mu = A_ill_mean, sigma = A_ill_sigma)
minuit_ill.migrad()
print(minuit_ill.fval)
#Ndof = 120-*minuit_ill.args
prob = stats.chi2.sf(minuit_ill.fval, (len(x)-len(minuit_ill.args)))
print (prob)

xaxis = np.linspace(0.005, 7, 1000)
yaxis = gauss_extended(xaxis, *minuit_ill.args)
names = ['Sum of t:','Entries','Mean','RMS', "Chi2/Ndof", "Prob"]
values = ["",
          "{:d}".format(len(t)),
          "{:.3f}".format(t.mean()),
          "{:.3f}".format(t.std(ddof=1)),
          "{0:.3f}/{1:.0f}".format(minuit_ill.fval,len(x)-len(minuit_ill.args)),
          "{:.3f}".format(stats.chi2.sf(minuit_ill.fval, (len(x)-len(minuit_ill.args))))
          ]
ax2.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax.transAxes, fontsize=10, verticalalignment='top')
ax2.set_xlabel("Sum of t")
ax2.set_ylabel("Frequency")
ax2.plot(xaxis, yaxis, '-', label='Fit to gaussian')
ax2.legend(loc = "best")
if SavePlots:
    fig2.savefig("sum_t.png")








plt.show(block=False)
try:
    __IPYTHON__
except:
    raw_input('Press Enter to exit')

#The end
