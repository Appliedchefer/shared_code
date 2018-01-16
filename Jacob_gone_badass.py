# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:08:34 2018

@author: Bruger
"""

# =============================================================================
# mix hitandmiss and trans
# =============================================================================

# Random numbers
Nnumbers = 10000
r = np.random
r.seed(42)
x_trans   = np.zeros(Nnumbers)
x_hitmiss = np.zeros(Nnumbers)

# Estimate the integral of exp(-x/3.0)*cos(x)^2 in the interval [0,inf] using a
# combination of Transformation and Hit-and-Miss method.


Nhit = 0
for i in range ( Nnumbers ) :
      
    # Well, the inside of the loop is up to you!
    
# Transformation method:
    # ----------------------
    # Integration gives the function F(x) = x^2, which inverted gives F^-1(x) = sqrt(x):
    # x_trans[i] = np.sqrt(r.uniform())      # ...so we let x_trans equal sqrt() of the uniform number!

    # Hit & Miss method:
    # ------------------
    # Generate two random numbers uniformly distributed in [0,1]x[0,2], until they
    # fulfill the "Hit requirement":
    
    x_hitmiss[i] = 0
    y = 1.1
    while (np.exp(-x_hitmiss[i])*pow(np.cos(x_hitmiss[i]),2) < y) :      # ...so keep making new numbers, until this is fulfilled!
        x_hitmiss[i] = -np.log(r.uniform())
        y  = np.exp(-x_hitmiss[i])*pow(np.cos(x_hitmiss[i]),2)
        Nhit += 1
        
       
Nbins = 50
Hist_HitMiss, Hist_HitMiss_edges = np.histogram(x_hitmiss, bins=Nbins)
Hist_HitMiss_centers = 0.5*(Hist_HitMiss_edges[1:] + Hist_HitMiss_edges[:-1])
Hist_HitMiss_error = np.sqrt(Hist_HitMiss)
Hist_HitMiss_indexes = Hist_HitMiss > 0     # Produce a new histogram, using only bins with non-zero entries

    
#---------------------------------------------------------------------------------- 
# Plot histograms on screen:
#---------------------------------------------------------------------------------- 

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlabel("Random number")
ax.set_ylabel("Frequency")
def fit(x, p0):
    return p0 * np.exp(-x)*np.cos(x)**2


chi2_object_hitmiss = Chi2Regression(fit, Hist_HitMiss_centers[Hist_HitMiss_indexes], Hist_HitMiss[Hist_HitMiss_indexes], Hist_HitMiss_error[Hist_HitMiss_indexes])
minuit_hitmiss = Minuit(chi2_object_hitmiss, pedantic=False)
minuit_hitmiss.migrad()
chi2_hitmiss = minuit_hitmiss.fval
ndof_hitmiss = chi2_object_hitmiss.ndof
prob_hitmiss = stats.chi2.sf(chi2_hitmiss, ndof_hitmiss)


# =============================================================================
# 
# names = ['Hit & Miss:','Entries','Mean','RMS','Chi2 / ndof','Prob','y-intercept','slope']
# values = ["",
#           "{:d}".format(len(x_hitmiss)),
#           "{:.3f}".format(x_hitmiss.mean()),
#           "{:.3f}".format(x_hitmiss.std(ddof=1)),
#           "{:.3f} / {:d}".format(chi2_hitmiss, ndof_hitmiss), 
#           "{:.3f}".format(prob_hitmiss),
#           "{:.3f} +/- {:.3f}".format(minuit_hitmiss.values['c'], minuit_hitmiss.errors['c']),
#           "{:.3f} +/- {:.3f}".format(minuit_hitmiss.values['m'], minuit_hitmiss.errors['m']),
#           ]
# ax.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax.transAxes, fontsize=10, verticalalignment='top')
# 
# =============================================================================
# Drawing histograms with errors in the same plot:
# =============================================================================
# x_fit = np.linspace(0, 10, 1000)
# y_fit = fit(x_fit, 1000)
# print(minuit_hitmiss.args)
# ax.plot(x_fit,y_fit)
# =============================================================================
ax.errorbar(Hist_HitMiss_centers[Hist_HitMiss_indexes], Hist_HitMiss[Hist_HitMiss_indexes], Hist_HitMiss_error[Hist_HitMiss_indexes], fmt='r.', capsize=2, capthick=2, label="Hit & Miss")
ax.set_ylim(bottom=0)

# Legend:
ax.legend(loc='lower right')

plt.tight_layout()
plt.show(block=False)

if (SavePlots) :
    fig.savefig("Hist_TransVsHitMiss.pdf", dpi=600)



# =============================================================================
# fkt to check for constant fit
# =============================================================================
def Check_if_constant(data, start, alpha):
    # Warning
    if len(data) == 0:
        print("Called function with an empty list")
        print("Thats an error")
        return "Bitch, you guessed it"
    # Warning
    if len(data) == 1:
        print("Fuck, called function with a list of length one")
        print("Thats an error")
        print(" ")
        return "Bitch, you guessed it"
    # Weighted mean
    if len(data) > 1:
        
        def func_cons(x,p0) :
                return p0
        data_cons = []
        data_cons.append(data[start])
        for j in range(1,len(data)):
            data_cons.append(data[j])
            
            
            
            x  = np.arange(0,len(data_cons))
            print(x)
            
            x  = np.asarray(x)
            y  = data_cons
            y  = np.asarray(y)
            print(y)
            sy = [0.1]*len(data_cons)
            sy = np.asarray(sy)
            mu = np.mean(data_cons)
            Chi2_cons = Chi2Regression(func_cons, x, y, sy)
            
            minuit_cons = Minuit(Chi2_cons, pedantic=False, p0=mu)#   
            minuit_cons.migrad()  # perform the actual fit
            
            chi2_cons = minuit_cons.fval
            ndof_cons = len(data_cons)-1
            
            prob_cons = stats.chi2.sf(chi2_cons, ndof_cons)
            p0 = minuit_cons.args
        
            p0 = minuit_cons.args[0]
            
            if prob_cons > alpha:
                #print("data is still consistant with constant for time: " + str(j+1))
            
                
            
                names = ['Chi2/ndf', 'Prob', 'Data fitted to mean (p0)']
                values = ["{:.3}/{:}".format(chi2_cons, ndof_cons), 
                  "{:.5f}".format(prob_cons), "{:.5}".format(p0)]
                
                fig, ax = plt.subplots(figsize=(12,6))
                ax.errorbar(x, y, yerr=sy, fmt='r.', ecolor='k', elinewidth=1, capsize=2, capthick=1)
                
                
                ax.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax.transAxes, fontsize=10, verticalalignment='top')
                plt.axhline(p0, min(x), max(x))
                ax.set_xlabel("time")
                ax.set_ylabel("unit")
                plt.title("Constant test")
                plt.legend()
                plt.tight_layout(pad=2.5)
                fig.savefig('constant_test.pdf', dpi=600)
            

        
    return prob_cons
data = [1.1,1.2,1.1,1.0,0.9,1.1,1.2,1.1,9,10,11,13,15,17,18,31,38,50,100,120,160,234]
alpha = 0.05
plot=True
start = 0
a = Check_if_constant(data, start, alpha)




# =============================================================================
# Bang og snavset er vaek
# =============================================================================


def Cillit_Bang(data, sigma,verbose):
    if verbose :
        data_array_before_anything = copy.copy(data)
        data_array1 = data_array_before_anything
        mean_rms = MeanAndRMS(data_array1)
        threshold = sigma
        remove = 0
        t = 0
        
        def func_Gaussian(x, N, mu, sigma) :
            return N * norm.pdf(x, mu, sigma)
        
        
        Nbins = 100
        counts, bin_edges = np.histogram(data_array1, bins=Nbins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
        s_counts = np.sqrt(counts)
        x = bin_centers[counts>0]
        y = counts[counts>0]
        sy = s_counts[counts>0]
        fig, ax = plt.subplots(figsize=(12,6))
        ax.errorbar(x, y, yerr=sy, fmt='r.', ecolor='k', elinewidth=1, capsize=2, capthick=1, label = "Histogram of cleaned data")
        plt.tight_layout(pad=2.5)
        
        Chi2_single = Chi2Regression(func_Gaussian, x, y, sy)
        minuit_single = Minuit(Chi2_single, pedantic=False, N=1, mu=mean_rms[1], sigma=mean_rms[3] )#   
        minuit_single.migrad()  # perform the actual fit
        
        chi2_Gaussian_s = minuit_single.fval
        ndof_Gaussian_s = Chi2_single.ndof
        prob_Gaussian_s = stats.chi2.sf(chi2_Gaussian_s, ndof_Gaussian_s)
        N, mu, sigma = minuit_single.args
        names = ["Cleaned Gaussian", '  Chi2/ndf', '  Prob', "Green Gaussian",'  Chi2/ndf', 'Prob', "N", "mu", "sigma"]
        values = ["", "{:.3f} / {:d}".format(chi2_Gaussian_s, ndof_Gaussian_s), 
          "{:.5f}".format(prob_Gaussian_s),
          "  {:.3f}".format(N),"  {:.3f}".format(mu),
          "  {:.3f}".format(sigma)]
        
        ax.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax.transAxes, fontsize=10, verticalalignment='top')
        
        ax.set_xlabel("sum")
        ax.set_ylabel("Frequency")
        plt.title("Distribution of the sum of 12 random numbers")   
        
        xaxis = np.linspace(min(data_array1), max(data_array1), 1000)
        yaxis_single = func_Gaussian(xaxis, *minuit_single.args)
        ax.plot(xaxis, yaxis_single, '-', label='Gaussian distribution (N, $\mu$, $\sigma$)')
        
        
        
        
        while (remove < maxgenerations):
            data_array_1less = []
            t += 1
            fucking_prob = [stats.norm.sf(np.abs(xtest - mean_rms[1]), loc = 0, scale = mean_rms[3]) for xtest in data_array1]
            #print (fucking_prob)
            #print (fucking_prob[np.argmin(fucking_prob)])
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
                
                mean_rms = MeanAndRMS(data_array1)
                if (verbose):
                    print("New calculated mean = {:6.4f} +- {:6.4f} m      RMS = {:6.4f} m   (N = {:3d})\n".format(
                            mean_rms[1], mean_rms[3], mean_rms[5], len(data_array1)))
            
            else:
                print ("\n\tAll datapoints are now above the threshold: {0:f} with the worst datapoint having a probability of: {1:.2E}".format(threshold, fucking_prob[np.argmin(fucking_prob)]))
                print("Values are: mean = {:6.4f} +- {:6.4f} m      RMS = {:6.4f} m   (N = {:3d})".format(
                            mean_rms[1], mean_rms[3], mean_rms[5], len(data_array1)))
                print ("And number of data removed is:", remove)
                break
        
        
        Nbins = int(round((max(data_array1)-min(data_array1))/(0.04*len(data_array1))))
        print(Nbins)
       
        counts, bin_edges = np.histogram(data_array1, bins=Nbins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
        s_counts = np.sqrt(counts)
        x = bin_centers[counts>0]
        y = counts[counts>0]
        sy = s_counts[counts>0]
        fig, ax = plt.subplots(figsize=(12,6))
        ax.errorbar(x, y, yerr=sy, fmt='r.', ecolor='k', elinewidth=1, capsize=2, capthick=1, label = "Histogram of cleaned data")
        plt.tight_layout(pad=2.5)
        
        Chi2_single = Chi2Regression(func_Gaussian, x, y, sy)
        minuit_single = Minuit(Chi2_single, pedantic=False, N=1, mu=mean_rms[1], sigma=mean_rms[3] )#   
        minuit_single.migrad()  # perform the actual fit
        
        chi2_Gaussian_s = minuit_single.fval
        ndof_Gaussian_s = Chi2_single.ndof
        prob_Gaussian_s = stats.chi2.sf(chi2_Gaussian_s, ndof_Gaussian_s)
        N, mu, sigma = minuit_single.args
        names = ["Cleaned Gaussian", '  Chi2/ndf', '  Prob', "Green Gaussian",'  Chi2/ndf', 'Prob', "N", "mu", "sigma"]
        values = ["", "{:.3f} / {:d}".format(chi2_Gaussian_s, ndof_Gaussian_s), 
          "{:.5f}".format(prob_Gaussian_s),
          "  {:.3f}".format(N),"  {:.3f}".format(mu),
          "  {:.3f}".format(sigma)]
        
        ax.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax.transAxes, fontsize=10, verticalalignment='top')
        
        ax.set_xlabel("sum")
        ax.set_ylabel("Frequency")
        plt.title("Distribution of the sum of 12 random numbers")   
        
        xaxis = np.linspace(min(data_array1), max(data_array1), 1000)
        yaxis_single = func_Gaussian(xaxis, *minuit_single.args)
        ax.plot(xaxis, yaxis_single, '-', label='Gaussian distribution (N, $\mu$, $\sigma$)')
    print ("Length after removing unlikely points:", len(data_array1))            
    return
               
        
        
Cillit_Bang(data, sigma_3, verbose=True)   