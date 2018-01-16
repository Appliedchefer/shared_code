from __future__ import division, print_function
import numpy as np
import uncertainties as unc
import uncertainties.unumpy as unumpy
from uncertainties import ufloat
from uncertainties.unumpy import exp
from uncertainties.umath import *
import matplotlib.pyplot as plt
from iminuit import Minuit
from probfit import BinnedLH, Chi2Regression, Extended, BinnedChi2, UnbinnedLH, gaussian
from scipy.special import erfc
from scipy import stats
from scipy.stats.stats import pearsonr   
import matplotlib.mlab as mlab
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn import mixture

import scipy

plt.close('all')

def mu_RMS (input_data = []):

    N = len(input_data)

    mu = np.sum(input_data)/N

    RMS = np.sqrt(np.sum((input_data-mu)**2)/(N-1))

    return mu, RMS

def GetMeanAndErrorOnMeanAndRMS( inlist = [] ) :
    """ Return list with mean and its error out of list of input values """
    if len( inlist ) == 0  :
        print("WARNING: Called function with an empty list")
        return [-9999999999.9, -99999999999.9]

    elif len( inlist ) == 1  :
        print("WARNING: Called function with a list of length one")
        return [inlist[0], -99999999999.9]

    # Find the mean:
    mu = 0.0
    for value in inlist :
        mu += value
    mu = float(mu) / len( inlist )

    # Find the standard deviation (rms) and error on mean (emu):
    rms = 0.0
    for value in inlist :
        rms += (value - mu)*(value - mu)
    rms = np.sqrt(float(rms) / (len(inlist) - 1))
    emu = float(rms) / np.sqrt(len(inlist))

    return [mu, emu, rms]


#### 1 ####
##########################################################################################################################
########################################################################################################################## 
# 1.1

print ("-------------- 1.1 --------------\n\n")
# 1.1
p1 = 1./6.
n1 = 4.0
p2 = 1./36.# probability of event 
n2 = 24.0 # total number 
first = stats.binom.pmf(1,n1,p1)
second = stats.binom.pmf(1,n2,p2)

total_second = 0.0
 #Starting probability (set to 0 so we can add onto it)
for i in range (1, 25): #In which range should the experiment count from? See example on p. 27 in Barlow
# remember that the range function does not take the last number into account!
    total_second += stats.binom.pmf(i,n2,p2) # For every i in range it calculates a probability and adds to the
    # previous probability.

total_first = 0.0
for i in range (1, 5):
    #print (i)
    total_first += stats.binom.pmf(i, n1, p1)
#print (total_second*100) #multiply by 100 to get it in percents.
print ("Second",total_second*100)

print("First", total_first*100)


##########################################################################################################################
########################################################################################################################## 
print ("-------------- 1.2 --------------\n\n")
lambda_mean = 19.3

test = stats.poisson.pmf(42, lambda_mean)
like = (((1-test))**1930)

print("Likelihood of seeing a 42 event in the entire period")
print (1-like)


##########################################################################################################################
##########################################################################################################################
print ("-------------- 1.3 --------------\n\n")
gauss_mean = 1.69
gauss_rms = 0.06

test_gauss = 1-stats.norm.cdf(1.85,gauss_mean, gauss_rms)

print ("Fraction of woman taller than 185: ",test_gauss)



times = 10000

gauss = np.random.normal(gauss_mean, gauss_rms, times)


#get the 80%

top_80 = np.percentile(gauss, 80)

print ("80 quantile - they should be hihger than this",top_80)

#append all the hights above 80% to a list and then calculate the average.
new_list = []
for i in range (len(gauss)):
    if gauss[i] > top_80:
        new_list.append(gauss[i])

print ("Numerically solved, the mean is:",sum(new_list)/len(new_list))
##########################################################################################################################
########################################################################################################################## 
print ("-------------- 2.1.1 --------------\n\n")

#This error propagation on error propagation in the sense that you need to take into
# account that r is squared. should be in relation L 1:2 r

##########################################################################################################################
########################################################################################################################## 
print ("-------------- 2.2.1 --------------\n\n")


m = 0.0084
sigma_m = 0.0005
velocity = np.array([361, 200, 390, 223, 232, 248, 243, 386, 443, 378])


mean_velocity, sigma_velocity, rms = GetMeanAndErrorOnMeanAndRMS(velocity)
print ("Get mean and emu and rms: ",mean_velocity,sigma_velocity,rms)
#mean_velocity, sigma_velocity = mu_RMS(velocity)

print("Avg speed: ",mean_velocity,"Sigma",sigma_velocity)

##########################################################################################################################
########################################################################################################################## 
print ("-------------- 2.2.2 --------------\n\n")
E_kin = 0.5 * m * mean_velocity**2
dEdm = 0.5*mean_velocity**2
dEdv = m* mean_velocity


sigma_E = np.sqrt(dEdm**2*sigma_m**2+dEdv**2*sigma_velocity**2)

print ("E_kin: ",E_kin,"sigma", sigma_E)

##########################################################################################################################
########################################################################################################################## 
print ("-------------- 2.2.3 --------------\n\n")
sigma_E_m_only = np.sqrt(dEdm**2*sigma_m**2)
sigma_E_velocity_only = np.sqrt(dEdv**2*sigma_velocity**2)

print ("m only: ",sigma_E_m_only,"V only", sigma_E_velocity_only)

#print ("meaning that sigma_m= sigma_v/sqrt(N), sqrt(N) = ",sigma_E_velocity_only/sigma_E_m_only," which is how much longer the list of experiments shoud be (9 now)")
print ((sigma_E_velocity_only/sigma_E_m_only)**2)


##########################################################################################################################
########################################################################################################################## 
print ("-------------- 3.1 --------------\n\n")

#Normalize the shit through integrals 

C = 0.243134





##########################################################################################################################
########################################################################################################################## 
print ("-------------- 4.1 --------------\n\n")
data = []
A_list = []
B_list = []
C_list = []
D_list = []

A_sick = []
B_sick = []
C_sick = []

A_nosick = []
B_nosick = []
C_nosick = []


with open( 'data_FisherSyndrome.txt', 'r' ) as infile :
    counter = 0

    for line in infile:
        line = line.strip().split()
        data.append([float(line[2]), float(line[3]), float(line[4]), int(line[0])])

        isIll = data[-1][3]
        A = data[-1][0]
        B = data[-1][1]
        C = data[-1][2]
        
        A_list.append(float(line[2]))
        B_list.append(float(line[3]))
        C_list.append(float(line[4]))
        D_list.append(float(line[0]))
        
        if float(line[0]) == 1.0:
            A_sick.append(float(line[2]))
            B_sick.append(float(line[3]))
            C_sick.append(float(line[4]))
        
        if float(line[0]) == 0.0:
            A_nosick.append(float(line[2]))
            B_nosick.append(float(line[3]))
            C_nosick.append(float(line[4]))
        
        
        # Print some numbers as a sanity check:
        #if (counter < 10) :
            #print("  Reading data:   Is the person ill? {0:d}   A = {1:5.2f}   B = {2:5.2f}   C = {3:5.2f}".format(isIll, A, B, C))
        #counter += 1
        
#y, b = gaussian_plot(A_sick,100)        
def gaussian_f(x, height, center, width):
    return height*np.exp(-(x - center)**2/(2*width**2))       


# do kolomogorov smornof
print (min(A_sick), max(A_sick))
mean = np.mean(A_sick)
var = np.var(A_sick)
sigma = (np.sqrt(var))
x = np.linspace(min(A_sick),max(A_sick),200)
bins = np.linspace(0,30,30)

hist,edges = np.histogram(A_sick,30)
fig,ax1 = plt.subplots(figsize=(5, 5))

ax1.hist(A_sick, edges, histtype='step',normed = True)
ax1.plot(x,mlab.normpdf(x,mean,sigma),"-r")


ax1.set_title("A_sick")
ax1.set_xlabel("Value")
ax1.set_ylabel('Frequency')

asd = Chi2Regression(gaussian_f,bins,hist) 

minuit = Minuit(asd, height = mean, width=sigma)
minuit.migrad()
minuit_output = [minuit.get_fmin(), minuit.get_param_states()]
ch2 = minuit.fval
prob = str(stats.chi2.sf(minuit.fval, len(A_sick)-3))
ax1.legend(["Fit","Measurement"],loc='upper right')
ax1.text(0.02, 0.95, ("Chi 2: 2005.6"+"\nProbability: 0.4417"), family='monospace', 
            transform=ax1.transAxes, fontsize=9, verticalalignment='top')
#print(ch2,prob,mean,sigma,"<------")


#print ("ejwhrewljhfkjhe",stats.kstest(mlab.normpdf(x,mean,sigma), 'norm', N = 100)) 


##########################################################################################################################
########################################################################################################################## 
# 4.1.2 - done 
print ("-------------- 4.1.2 --------------\n\n")
B_sick = np.asarray(B_sick)
C_sick = np.asarray(C_sick)
fig2,ax2 = plt.subplots(figsize=(5, 5))

ax2.plot(B_sick,C_sick,".")
fit = np.polyfit(B_sick, C_sick, deg=1)
ax2.plot(B_sick, fit[0] * B_sick + fit[1], color='red')
coeff,prob =  (pearsonr(B_sick,C_sick))
ax2.set_title("B vs C correlation")
ax2.set_xlabel("B")
ax2.set_ylabel('C')
ax2.legend(["Measurements","Fit"],loc='upper right')
ax2.text(0.02, 0.95, ("Pearsons correlation coef: "+str(format(coeff,'.2f'))+"\nProbability: "+str(format(prob,'.2f'))), family='monospace', 
            transform=ax2.transAxes, fontsize=9, verticalalignment='top')



def gaussian_plot(data,bins2): # give data and bins and returns a y value
    hist,bins = np.histogram(data,bins2, normed = True)
    mu, sigma = norm.fit(data)
    y =mlab.normpdf(bins,mu,sigma)
    return y,bins
def chi2_calc(y,ey,known):
    chi2 = 0.
    for i in range(len(y)):
        chi2 += ((y[i]-known)/((ey[i])**2))**2
    return chi2
    


##########################################################################################################################
########################################################################################################################## 
# 4.1.3
    print ("-------------- 4.1.3 --------------\n\n")
def f_val(A,B,C):
    return (-1.33*A + 0.63*B+2.1*C)


#creating lists of values:
F_nosick = []
for i in range(len(A_nosick)):
    F_nosick.append(f_val(A_nosick[i],B_nosick[i],C_nosick[i]))
F_sick = []
for i in range(len(A_sick)):
    F_sick.append(f_val(A_sick[i],B_sick[i],C_sick[i]))
    
fig3,ax3 = plt.subplots(figsize=(5, 5))  

ax3.hist(F_nosick, 100, histtype='step', color ="g",normed = True)

y,bins = gaussian_plot(F_nosick,100)
ax3.plot(bins,y,"b--")


ax3.hist(F_sick, 100, histtype='step', color ="r",normed = True)
y,bins = gaussian_plot(F_sick,100)
ax3.plot(bins,y,"r--")
ax3.set_title("Test statistics for F")
ax3.set_xlabel("Value")
ax3.set_ylabel('Frequency')
ax3.legend(["Fit Not sick","Fit sick","Not sick", "Sick"],loc='upper left')

# find sigma and mu
mu_nosick,sigma_nosick = norm.fit(F_nosick)
mu_sick,sigma_sick = norm.fit(F_sick)

percent_typeI_errors = norm.sf(20,loc = mu_sick, scale = sigma_sick)
percent_typeII_errors = norm.cdf(20,loc = mu_nosick, scale = sigma_nosick)
# setting criteria at 20 - ask what the others did, check up on the errors being true

#print(percent_typeI_errors)


thr = norm.ppf(0.999,loc= mu_sick,scale = sigma_sick)
print (thr)

percent_typeII_errors = norm.cdf(thr,loc = mu_nosick, scale = sigma_nosick) 
print(percent_typeII_errors)


#---------------------------------------------------------------------------------- 
# Your analysis...
#--------------------------------------------------------------------------



# Define input data lists:
x   = []
y   = []
ex  = []
ey  = []

with open( 'data_LukeLightningLights.txt', 'r' ) as infile :
    for line in infile:
        line = line.strip().split()
        x.append(float(line[0]))
        ex.append(float(line[1]))
        y.append(float(line[2]))
        ey.append(float(line[3]))

        # Print the numbers as a sanity check (only 41 in total):
        #print("  Read data:  {0:6.3f}  {1:6.3f}  {2:6.3f}  {3:6.3f}".format(x[-1], ex[-1], y[-1], ey[-1]))

##########################################################################################################################
########################################################################################################################## 
#5.1.1
print ("-------------- 5.1.1 --------------\n\n")
x_first_12   = np.asarray(x[0:12])
y_first_12   = np.asarray(y[0:12])
ey_first_12  = np.asarray(ey[0:12])

def fit_function(x,alpha0,alpha1):
    return alpha0+alpha1*x

chi2_object = Chi2Regression(fit_function, x_first_12, y_first_12, ey_first_12)
minuit = Minuit(chi2_object, pedantic=False, alpha0=np.mean(y_first_12),alpha1 = 0,fix_alpha0 = True, print_level=0) # ,fix_alpha1= True for the first 
minuit.migrad();
minuit_output = [minuit.get_fmin(), minuit.get_param_states()] # save the output parameters in case needed
    
alpha0_fit = minuit.values['alpha0']
alpha1_fit = minuit.values['alpha1']
sigma_alpha0_fit = minuit.errors['alpha0']
sigma_alpha1_fit = minuit.errors['alpha1']


Chi2_fit = minuit.fval # the chi2 value
Prob_fit =  stats.chi2.sf(Chi2_fit, len(y_first_12)-1)


#chi2_first_12 = chi2_calc(y_first_12,ey_first_12,np.mean(y_first_12))
#prob_first_12 =stats.chi2.sf(chi2_first_12,11) # ask the others about this

print ("Constant:",alpha0_fit,"\nChi2:",Chi2_fit,"\nProb:",Prob_fit)


# find mean - ask the others about this, and calculate chi2 from this - below ss best fit

##########################################################################################################################
########################################################################################################################## 
# 5.1.2
print ("-------------- 5.1.2 --------------\n\n")
def fit_function(x,alpha0,alpha1):
    return alpha0+alpha1*x

chi2_object = Chi2Regression(fit_function, x_first_12, y_first_12, ey_first_12)
minuit = Minuit(chi2_object, pedantic=False, alpha0=-0.4, alpha1=0, print_level=0) # ,fix_alpha1= True for the first 
minuit.migrad();
minuit_output = [minuit.get_fmin(), minuit.get_param_states()] # save the output parameters in case needed
    
alpha0_fit = minuit.values['alpha0']
alpha1_fit = minuit.values['alpha1']
sigma_alpha0_fit = minuit.errors['alpha0']
sigma_alpha1_fit = minuit.errors['alpha1']

# In Minuit, you can just ask the fit function for it:
Chi2_fit = minuit.fval # the chi2 value
Prob_fit =  stats.chi2.sf(Chi2_fit, len(y_first_12)-2)

#print(alpha0_fit,alpha1_fit,Chi2_fit,Prob_fit)





fig4,ax4 = plt.subplots(figsize=(5, 5))
ax4.errorbar(x_first_12, y_first_12, ey_first_12,fmt='none', ecolor='k', elinewidth=1, capsize=2, capthick=1)
ax4.plot(x_first_12,y_first_12,".")
ax4.plot(x_first_12,fit_function(x_first_12,*minuit.args),'r-')
ax4.set_title("Income first 12 months")
ax4.set_xlabel("month #")
ax4.set_ylabel('income')
ax4.legend(["Measurements","fit"],loc='lower right')

ax4.text(0.02, 0.95, ("Fit: "+str(format(alpha0_fit,'.2f'))+"+"+str(format(alpha1_fit,'.2f'))+"*x"+"\nProb: "+str(format(Prob_fit,'.2f'))+"\nChi2: "+str(format(Chi2_fit,'.2f'))), family='monospace', 
            transform=ax4.transAxes, fontsize=9, verticalalignment='top')

x = np.asarray(x)
# how long maintained
threshold = 0.01


for i in range(12,len(x)):
    
    x_here = np.asarray(x[0:i])
    y_here = np.asarray(y[0:i])
    ey_here = np.asarray(ey[0:i])
        
    chi2_object = Chi2Regression(fit_function, x_here, y_here, ey_here)
    minuit = Minuit(chi2_object, pedantic=False, alpha0=-0.4, alpha1=0, print_level=0) # ,fix_alpha1= True for the first 
    minuit.migrad();
    minuit_output = [minuit.get_fmin(), minuit.get_param_states()] # save the output parameters in case needed
    
    alpha0_fit = minuit.values['alpha0']
    alpha1_fit = minuit.values['alpha1']
    sigma_alpha0_fit = minuit.errors['alpha0']
    sigma_alpha1_fit = minuit.errors['alpha1']
    
    # In Minuit, you can just ask the fit function for it:
    Chi2_fit = minuit.fval # the chi2 value
    Prob_fit =  stats.chi2.sf(Chi2_fit, len(y_here)-2)
    
    if Prob_fit <0.01:
        print ("Probability below threshold of 0.01","\nProb: ", Prob_fit,"Last month where hypothesis is okay",i-1)
        break

##########################################################################################################################
##########################################################################################################################        
# 5.1.3
print ("-------------- 5.1.3 --------------\n\n")
      
    
  
    
def sigmoid(x, x0, k, x4,neg):
     y =  1 / (1 + np.exp(-k*(x-x0)))
     return neg + y*x4     
def sigmoidal2(x, L, k, x0, C1, C2):
   return (L / (1 + np.exp(-k * x - x0))) - C1 * (x <= 31) + C2 * (x > 31)


x= np.asarray(x)
y= np.asarray(y)
ey= np.asarray(ey)

chi2_object3 = Chi2Regression(sigmoidal2, x, y, ey)
minuit3 = Minuit(chi2_object3, pedantic=False,C2= 0.1,C1=0, print_level=0) # ,fix_alpha1= True for the first 
minuit3.migrad();
minuit3_output = [minuit3.get_fmin(), minuit3.get_param_states()]
  
# sigmoidal tryhard - for fitting initial part

x_here= np.asarray(x[0:31])
y_here= np.asarray(y[0:31])
ey_here = np.asarray(ey[0:31])

# for fitting the last part
x_last = np.asarray(x[31:len(x)])
y_last = np.asarray(y[31:len(x)])
ey_last = np.asarray(ey[31:len(x)])



#try minuit sigmoidal first part
chi2_object = Chi2Regression(sigmoid, x_here, y_here, ey_here)
minuit = Minuit(chi2_object, pedantic=False, neg = -0.34,print_level=0) # ,fix_alpha1= True for the first 
minuit.migrad();
minuit_output = [minuit.get_fmin(), minuit.get_param_states()]

print("---->",y_last[0])



x0_fit = minuit.values['x0']
k_fit = minuit.values['k']
x4_fit = minuit.values['x4']

sigma_x0_fit = minuit.errors['x0']
sigma_k_fit = minuit.errors['k']
sigma_x4_fit = minuit.errors['x4']

Chi2_fit_sigmoidal = minuit.fval # the chi2 value
Prob_fit_sigmoidal =  stats.chi2.sf(Chi2_fit_sigmoidal, len(x_here)-4)

# minuit linar last part
chi2_object2 = Chi2Regression(fit_function, x_last, y_last, ey_last)
minuit2 = Minuit(chi2_object2, pedantic=False, print_level=0) # ,fix_alpha1= True for the first 
minuit2.migrad();
minuit2_output = [minuit2.get_fmin(), minuit2.get_param_states()]

alpha0_fit = minuit2.values['alpha0']
alpha1_fit = minuit2.values['alpha1']
sigma_alpha0_fit = minuit2.errors['alpha0']
sigma_alpha1_fit = minuit2.errors['alpha1']


Chi2_fit_linear = minuit2.fval # the chi2 value
Prob_fit_linear =  stats.chi2.sf(Chi2_fit_linear, len(y_last)-2)

fig5,ax5 = plt.subplots(figsize=(5, 5))
ax5.errorbar(x, y, ey,fmt='none', ecolor='k', elinewidth=1, capsize=2, capthick=1)
ax5.plot(x,y,".")
ax5.plot(x_here,sigmoid(x_here,*minuit.args),'r-')
ax5.plot(x_last,fit_function(x_last,*minuit2.args),'b-')
#ax5.plot(x,sigmoidal2(x,*minuit3.args),'g-') # include the full fit

ax5.set_title("Income all months")
ax5.set_xlabel("month #")
ax5.set_ylabel('income')
ax5.legend(["Measurements","fit sigmoidal","fit linear"],loc='lower right')  
ax5.text(0.02, 0.95, ("Sigmoidal Chi2: "+str(format(Chi2_fit_sigmoidal,'.2f'))+"\nSigmoidal Prob: "+str(format(Prob_fit_sigmoidal,'.2f'))+"\nLin Chi2: "+str(format(Chi2_fit_linear,'.2f'))+"\nLin Prob: "+str(format(Prob_fit_linear,'.2f'))), family='monospace', 
            transform=ax5.transAxes, fontsize=9, verticalalignment='top')
   
    
#estimate drop - ask the others
##########################################################################################################################
########################################################################################################################## 
    
    



data = []
counter = 0

with open( "data_TimingResiduals.txt", 'r' ) as infile :
    for line in infile:
        line = line.strip().split()
        data.append(float(line[0]))
        #if (counter < 10) :
            #print("  {0:4d}:    {1:6.3f} seconds".format(counter, data[-1]))
        counter += 1

#print("  Number of measurements in total: ", counter)

timing_all = np.asarray(data)


##########################################################################################################################
########################################################################################################################## 
print ("-------------- 5.2.1 --------------\n\n")


mu, rms = mu_RMS(timing_all)

print (mu,rms)


print (scipy.stats.ttest_1samp(timing_all, 0))
# overall is consistent, however high rms!!!

##########################################################################################################################
########################################################################################################################## 
#5.2.2
print ("-------------- 5.2.2 --------------\n\n")
fig6,ax6 = plt.subplots(figsize=(5, 5))
ax6.plot(timing_all,'.')


ax6.set_title("All timing residuals")
ax6.set_xlabel("measurement #")
ax6.set_ylabel('Value')


fig7,ax7 = plt.subplots(figsize=(5, 5))
ax7.hist(timing_all, 100, histtype='step', color ="r",normed = True)




y, bins = gaussian_plot(timing_all,100)
ax7.plot(bins,y,"r--")



#obviuosly some persons are a lot off ? not sure whats meant by the question  - could remove some of the worst - but easier to answer after next part

##########################################################################################################################
########################################################################################################################## 
#5.2.3
print ("-------------- 5.2.2 --------------\n\n")


#y,bins = np.histogram(timing_all,50m normed = True)
#x_here = np.linspace(-1,1,50)

"""
chi2_object = Chi2Regression(gaussian_f, bins, y)
minuit = Minuit(chi2_object, pedantic=False,height= 7,width = rms,center= mu,print_level=0) # ,fix_alpha1= True for the first 
minuit.migrad();
minuit_output = [minuit.get_fmin(), minuit.get_param_states()]

Chi2_fit = minuit.fval # the chi2 value
Prob_fit =  stats.chi2.sf(Chi2_fit, len(bins)-3)

fig8,ax8 = plt.subplots(figsize=(5, 5))
ax8.hist(timing_all, 100, histtype='step', color ="g",normed = True)
ax8.plot(bins, gaussian_f(bins,*minuit.args),'--r')
ax8.set_title("Binned timing residuals and gaussian fits")
ax8.set_xlabel("Value")
ax8.set_ylabel('Frequency')
ax8.legend(["Residuals","fit"],loc='upper right')  
ax8.text(0.02, 0.95, ("Gaussian Chi2: "+str(format(Chi2_fit,'.2f'))+"\nGaussian Prob: "+str(format(Prob_fit,'.2f'))), family='monospace', 
            transform=ax8.transAxes, fontsize=9, verticalalignment='top')
plt.close()
"""
##########################################################################################################################
########################################################################################################################## 
#5.2.3



resids = timing_all



obs = timing_all.reshape(-1, 1)

min_rng = -0.4
max_rng = 0.4

#plt.hist(obs, bins = 100, normed = True, color = "lightgrey", label = "Raw data", range = (min_rng, max_rng))

n_gaussians = 2
m = mixture.BayesianGaussianMixture(n_components = n_gaussians,
                                    mean_prior = np.array([0]),
                                    mean_precision_prior = np.array([1]),
                                    max_iter = 1000)

m.means_init = []
m.fit(obs)

# # Get the gaussian parameters
weights = m.weights_
means = m.means_
covars = m.covariances_

gaussian_sum = []

for i in range(n_gaussians):

    mu = means[i]
    sigma = np.sqrt(covars[i]).reshape(1)

    plotpoints = np.linspace(min_rng, max_rng, 1000)

    gaussian_points = weights[i] * stats.norm.pdf(plotpoints, mu, sigma)
    ks_dist = stats.norm.rvs(mu, sigma, int(np.asscalar(weights[i]))*5000)

    gaussian_points = np.array(gaussian_points)

    gaussian_sum.append(gaussian_points)

    plt.plot(plotpoints, gaussian_points, label = "Gaussian component {}".format(i+1))

sum_gaussian = np.sum(gaussian_sum, axis=0)
plt.plot(plotpoints, sum_gaussian, color = "black", linestyle = "--", label = "Summed gaussian fit")
plt.xlim(min_rng, max_rng)

# Generate mixture distribution from obtained parameters (WHY DO I NEED SQRT HERE BUT NOT ABOVE????)
a = stats.norm.rvs(3.20e-3, np.sqrt(1.49e-2), int(5000*0.46))
b = stats.norm.rvs(-2.50e-3, np.sqrt(1.56e-3), int(5000*0.54))

mixdist = np.concatenate([a, b])
#plt.hist(mixdist, normed = True, bins = 100, histtype = "step", color = "firebrick", label = "Sim dist from fit params", range = (min_rng, max_rng))

# Calculate the KS test score
KS_D, KS_p = stats.ks_2samp(resids, mixdist)


plt.text(x = -0.37, y = 6.5, s = "Kolmogorov-Smirnoff test:\nD = {0:.2f}\np = {1:.2f}".format(KS_D, KS_p))
"""
plt.text(x = -0.37, y = 3, s = "$\mu_1$ = ${}$\n"
                               "$\sigma_1$ = ${}$\n"
                               "$w_1$ = ${:.2f}$\n\n"
                               "$\mu_2$ = ${}$\n"
                               "$\sigma_2$ = ${}$\n"
                               "$w_2$ = ${:.2f}$".format(print_sci(float(means[0]),decimals = 2),
                                                         print_sci(np.sqrt(float(covars[0])), decimals = 2),
                                                         weights[0],
                                                         print_sci(float(means[1]), decimals = 2),
                                                         print_sci(np.sqrt(float(covars[1])), decimals = 2),
                                                         weights[1]))
"""

print(means[0],
covars[0],
means[1],
covars[1])
plt.legend()
plt.tight_layout()
plt.show()


"""








































#my own
print ("-------------- 5.2.3 --------------\n\n")
def gauchy(x,scal,height,center):
    return height*(1/ (np.pi*scal))*(scal**2)/((x-center)**2+scal**2)

chi2_object = Chi2Regression(gauchy, bins, y)
minuit = Minuit(chi2_object, pedantic=False,scal=0.01,center= mu,height = 6,print_level=0) # ,fix_alpha1= True for the first 
minuit.migrad();
minuit_output = [minuit.get_fmin(), minuit.get_param_states()]
Chi2_fit = minuit.fval # the chi2 value
Prob_fit =  stats.chi2.sf(Chi2_fit, len(bins)-3)


above = []
below = []

for i in range(len(timing_all)):
    if abs(timing_all[i]) >rms:
        above.append(timing_all[i])
    if abs(timing_all[i]) <=rms:
        below.append(timing_all[i])

fig9,ax9 = plt.subplots(figsize=(8, 5))
ax9.hist(timing_all, 100, histtype='step', color ="g",normed = True)

#ax9.plot(bins, gauchy(bins,*minuit.args),'--r')
#ax9.text(0.02, 0.95, ("Sigmoidal Chi2: "+str(format(Chi2_fit,'.2f'))+"\nSigmoidal Prob: "+str(format(Prob_fit,'.2f'))), family='monospace', 
            #transform=ax9.transAxes, fontsize=9, verticalalignment='top')
print (Chi2_fit,Prob_fit,"sdas")





def gaussian_f2(x, height, center, width,height2,center2,width2):
    return gaussian_f(x,height,center,width)+gaussian_f(x,height2,center2,width2)

#height*np.exp(-(x - center)**2/(2*width**2)) + (height2*np.exp(-(x - center2)**2/(2*width2**2)))
                           
chi2_object = Chi2Regression(gaussian_f2, bins, y) 

minuit = Minuit(chi2_object, pedantic=False, height = 2,center= mu, width = rms/12, height2 =1, center2 = mu, width2 = rms*7,print_level=0) # ,fix_alpha1= True for the first 
minuit.migrad();
minuit_output = [minuit.get_fmin(), minuit.get_param_states()]
Chi2_fit = minuit.fval # the chi2 value
Prob_fit =  stats.chi2.sf(Chi2_fit, len(bins)-6)

#print (Chi2_fit,Prob_fit,"sdas")

#ax9.plot(bins, gaussian_f(bins,*minuit.args[0:3]),'--b')
#ax9.plot(bins, gaussian_f(bins,*minuit.args[3:6]),'--b')

#ax9.plot(bins,(gaussian_f(bins,*minuit.args[0:3]))+gaussian_f(bins,*minuit.args[3:6]), '--g')


resids = timing_all



obs = timing_all.reshape(-1, 1)

min_rng = -0.4
max_rng = 0.4

#plt.hist(obs, bins = 100, normed = True, color = "lightgrey", label = "Raw data", range = (min_rng, max_rng))

n_gaussians = 2
m = mixture.BayesianGaussianMixture(n_components = n_gaussians,
                                    mean_prior = np.array([0]),
                                    mean_precision_prior = np.array([1]),
                                    max_iter = 1000)

m.means_init = []
m.fit(obs)

# # Get the gaussian parameters
weights = m.weights_
means = m.means_
covars = m.covariances_

gaussian_sum = []

for i in range(n_gaussians):

    mu = means[i]
    sigma = np.sqrt(covars[i]).reshape(1)

    plotpoints = np.linspace(min_rng, max_rng, 1000)

    gaussian_points = weights[i] * stats.norm.pdf(plotpoints, mu, sigma)
    ks_dist = stats.norm.rvs(mu, sigma, int(np.asscalar(weights[i]))*5000)

    gaussian_points = np.array(gaussian_points)

    gaussian_sum.append(gaussian_points)

    plt.plot(plotpoints, gaussian_points, label = "Gaussian component {}".format(i+1))

sum_gaussian = np.sum(gaussian_sum, axis=0)
ax9.plot(plotpoints, sum_gaussian, color = "black", linestyle = "--")


ax9.set_title("Double gaussian fit for residuals")
ax9.set_xlabel("value")
ax9.set_ylabel('Frequency')
ax9.legend(["Gaussian 1","Gaussian 2","Sum Gaussian 1+2","Residuals"],loc='upper right')
print (stats.ks_2samp(timing_all,sum_gaussian))

# Generate mixture distribution from obtained parameters (WHY DO I NEED SQRT HERE BUT NOT ABOVE????)
a = stats.norm.rvs(3.20e-3, np.sqrt(1.49e-2), int(5000*0.46))
b = stats.norm.rvs(-2.50e-3, np.sqrt(1.56e-3), int(5000*0.54))

mixdist = np.concatenate([a, b])
#plt.hist(mixdist, normed = True, bins = 100, histtype = "step", color = "firebrick", label = "Sim dist from fit params", range = (min_rng, max_rng))

# Calculate the KS test score
KS_D, KS_p = stats.ks_2samp(resids, mixdist)
print(KS_D,KS_p)
ax9.text(0.02, 0.95, ("kolmogorov S. test: "+str(format(KS_D,'.2f'))+"\nK.S test Prob: "+str(format(KS_p,'.2f'))), family='monospace', 
            transform=ax5.transAxes, fontsize=9, verticalalignment='top')

"""