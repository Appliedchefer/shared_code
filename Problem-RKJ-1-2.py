#----------------------------------------------------------------------------------
# Rasmus:
#----------------------------------------------------------------------------------
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import math
from iminuit import Minuit
from probfit import Chi2Regression, BinnedLH
from scipy import stats
plt.close('all')

r = np.random                       # Random generator
r.seed(42)                          # Fixed order of random numbers

Showgraph = True                    # Need to be True for Saveplots to work
SavePlots = False
verbose = True
Nverbose = 10


#----------------------------------------------------------------------------------
# Assignments:
#----------------------------------------------------------------------------------
print ("DISTRIBUTIONS AND PROBABILITIES - 1")
print ("Assigment 1.1:")

#For the game with 1 dice and 4 rolls
dice_range_1 = range(1, 4+1)  #Succes we expect - given as 1 to 4 - win if 1, 2, 3 or 4 sixers
dice_prob_1 = 0.0
for i in dice_range_1: 
    dice_prob_1 += stats.binom.pmf(i, 4, 1./6.) #pmf(successes, number of tries (large N), probability)
    #print ("The or pmf", stats.binom.pmf(i, 4, 1./6.))
print ("The one dice game probability of winning:", dice_prob_1)


#For the game with 2 dice and 24 rolls
dice_range_2 = range(1, 24+1)  #Succes we expect - given as 1 to 24 
dice_prob_2 = 0.0
for i in dice_range_2: 
    dice_prob_2 += stats.binom.pmf(i, 24, 1./36.)
    #print ("for pmf", stats.binom.pmf(i, 4, 1./6.))
print ("The two dice game probability of winning:", dice_prob_2)



#The code below is the handwritten version of pmf: Both give the same results
"""
def probability_binomial(r, p, n):
    return p**r * (1-p)**(n-r) * (math.factorial(n)/(math.factorial(r) * math.factorial(n-r)))

p = 1./6.        #Probability for succes
n = 4            #Trials
r = range(1, n+1)  #Succes we expect - given 
probability = 0.0
for i in r:
    probability += probability_binomial(i, p, n)
    print ("For", i, "the combined probability is:", probability) 
    uu = probability_binomial(i, p, n)
    print ("With the individual contribution of:", uu)
print ("Probability 1x6:", probability)

#For the second example:
p = 1./36.        #Probability for succes
n = 24            #Trials
r = range(1, n+1)  #Succes we expect
probability = 0.0
for i in r:
    probability += probability_binomial(i, p, n)
print ("Probability 2x6:",probability)
"""





#----------------------------------------------------------------------------------
# Assignments:
#----------------------------------------------------------------------------------
print ("\n\n-----------------------------")
print ("Assigment 1.2:")

mean = 19.3
#Plot for fun
if (Showgraph):   
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(stats.poisson.ppf(0.001, mean), stats.poisson.ppf(0.999, mean))
    ax.plot(x, stats.poisson.pmf(x, mean), '-', ms=8, label='poisson pmf')
    ax.set_title('Assignemnt 1.2 - Poisson')
    ax.set_xlabel('Events')
    ax.set_ylabel('Percentages')
    
    y = [0*t for t in range(x.shape[0])]
    ax.plot(x, y, 'r-', label='Probability = 0')
    plt.legend()
    fig.tight_layout()
    plt.show(block=False)
    if (SavePlots): 
        fig.savefig("figures/Assignment-12.pdf")
    

all_days = 1930
print ("Finding probability for 42 events when mean is {:.1f} for one day:".format(mean))
print (stats.poisson.pmf(42, mean)) #If you times with days you can get a number larger than 1 so no good. 
#If you take to the power, then it gives zero, and not true. Therefore do the following:

print ("Probability of not happening is 1 minus that it happens:")
not_happen_1day = 1 - stats.poisson.pmf(42, mean)
print (not_happen_1day)

print ("Probability for not happening over all days (to the power of all days)")
not_happen_all_days = not_happen_1day**(all_days) 
print (not_happen_all_days)

print ("Probability of happening over all days (1 minus the last number)")
happen_all_days = 1 - not_happen_all_days
print (happen_all_days)
print ("Which is around 2 to 3 sigma from mean which is ok (Almost Gaussian).")






#----------------------------------------------------------------------------------
# Assignments:
#----------------------------------------------------------------------------------
print ("\n\n-----------------------------")
print ("Assigment 1.3:")
mean_height = 1.69
std_height = 0.06

print ("Fraction of woman taller than 1.85 m:")
tall_woman = 1-stats.norm.cdf(1.850001, mean_height, std_height)
print (tall_woman)

#Plot of the Gaussian distribution
if (Showgraph):   
    Nexperiments = 10000
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    #If we want random experiments:
    #Nbins = 300
    #ax1.hist(woman_gaussian, bins=Nbins, histtype='step', linewidth=2,label='Gaussian ($\mu$ = 1.69)', color='blue')
    
    #If the real distributions with know values: 
    x = np.linspace(stats.norm.ppf(0.0001, mean_height, std_height), 
                    stats.norm.ppf(0.99999, mean_height, std_height), Nexperiments)
    ax1.plot(x, stats.norm.pdf(x, mean_height, std_height), 'r-', label='Gaussian ($\mu$ = 1.69 m and $\sigma$ = 0.06 m)')
    ax1.set_title('Assignemnt 1.3 - Women height with a Gaussian PDF')
    ax1.set_xlabel('Heights')
    ax1.set_ylabel('Gaussian - Probability density function')
    ax1.set_ylim(0, 7)
    plt.legend()
    fig1.tight_layout()
    plt.show(block=False)
    if (SavePlots): 
        fig1.savefig("figures/Assignment-13.pdf")




print ("\nTo find the numerically average height of the 20% tallest women:")
smallest_20_height = stats.norm.ppf(0.80, mean_height, std_height)
print ("Height of smallest of the last 20 %", smallest_20_height)
pp = stats.norm.sf(smallest_20_height, 1.69, 0.06)
print ("to see it is 20 % and up", pp)


print ("To find the average height of the tallest 20 % women numerically:")
Nexperiments = 10000
tallest_20_woman = []

woman_gaussian = np.random.normal(1.69, 0.06, Nexperiments)
for i in range(len(woman_gaussian)):
    if (1-stats.norm.cdf(woman_gaussian[i], 1.69, 0.06)) < 0.20: 
        tallest_20_woman.append(woman_gaussian[i])
    
for i in range(len(tallest_20_woman)):
    if (verbose and i < Nverbose and len(tallest_20_woman) > 10): 
        print ("  10 height of 20 % tallest women:{:4f}".format(tallest_20_woman[i]))

mean_tallest_20 = np.mean(tallest_20_woman)
print ("The mean of these 20 % is: {:.3}".format(mean_tallest_20))


#If curious - plot of the tallest 20 % women
"""    
fig, ax = plt.subplots(figsize=(12, 4))
ax.hist(tallest_20_woman)
plt.legend()
fig.tight_layout()
plt.show(block=False)
"""




#----------------------------------------------------------------------------------
# Assignments:
#----------------------------------------------------------------------------------
print ("\n\n-----------------------------")
print ("ERROR PROPAGATION - 2")
print ("Assigment 2.1:")
print ("\n\nTEST\n\n")

#The equation: 
#R = L/A where A = pi * r^2
#The relation is the double due r being to the power of 2: 

#NOT THE SAME!
test = np.sqrt((1/1000)**2 + 2*(1/2000)**2)
print (test)
print (np.sqrt((1/1000)**2))
print (np.sqrt(2*(1/2000)**2))




#----------------------------------------------------------------------------------
# Assignments:
#----------------------------------------------------------------------------------
print ("\n\n-----------------------------")
print ("Assigment 2.2:")

measurement = [i for i in range(1, 11)]
results = [3.61, 2.00, 3.90, 2.23, 2.32, 2.48, 2.43, 3.86, 4.43, 3.78]
print (len(results))
results = [i * 10**2 for i in results]

print ("Part 1")
speed_average  = np.mean(results)
speed_uncertainty = (np.std(results, ddof = 1))/np.sqrt(len(results)) #1 degree of freedom because we dont know mean!
print ("Average spped:", speed_average)
print ("Uncertainty on speed:", speed_uncertainty)



print ("\nPart 2")
bullet_mass = 8.4*10**(-3)
bullet_uncertainty = 0.5 *10**(-3)


#Equation: Ekin = 1/2 * m * v**2, and use the law of combination of errors for independent variables (page 57)
Ekin = 1./2. * bullet_mass*(speed_average)**2 

dEdm = 1./2. * (speed_average)**2                           #Diff with respect to m: 1/2 v**2
err_E_fromm = dEdm * (bullet_uncertainty)                   #err_E_fromm = dEdm * sigma_m

dEdv = bullet_mass * speed_average                          #Diff with respect to v: m*v
err_E_fromv = dEdv * speed_uncertainty                      #err_g_fromv = dEdv * sigma_v

err_E = np.sqrt( (err_E_fromm)**2 + (err_E_fromv)**2 )      #Combined error - quadrature

print("The result: Ekin = {0:6.4f} +- {1:6.4f} (m) +- {2:6.4f} (v) = {3:6.4f} +- {4:6.4f}".format(Ekin, err_E_fromm, err_E_fromv, Ekin, err_E))


print ("\nPart 3")
print ("For each uncertainty, see above!")

print ("Calculate N:")
#from: err_E_fromm = err_E_fromv / sqrt(10 + x) and we would like to find x
N_more_measurements = (err_E_fromv/err_E_fromm)**2
print (N_more_measurements)
print ("And this is with 10 measurements, so times 10")
print (N_more_measurements * 10)




#Test if correct
"""
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

mu, emu, std = GetMeanAndErrorOnMean(results)
print (mu, emu, std )
"""



















#
