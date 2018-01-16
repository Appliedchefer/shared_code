from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from probfit import BinnedLH, Chi2Regression, Extended, cauchy
from scipy.special import erfc
from scipy import stats

SavePlots = True

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

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def gauss_extended(x, N1, mu1, sigma1, N2, mu2, sigma2) :
    """Non-normalized Gaussian"""
    return N1 * gauss_pdf(x, mu1, sigma1) + N2 * gauss_pdf(x, mu2, sigma2)

def gauss_extended_norm(x, N, mu, sigma) :
    """Non-normalized Gaussian"""
    return N * gauss_pdf(x, mu, sigma)


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

data = np.array(data)

mean_res, sigma_res, err_mean_res = mu_RMS(data)

print (mean_res, sigma_res, err_mean_res)

fig, ax = plt.subplots (figsize = (10,5))

counts_res, bin_edges_res = np.histogram(data, bins=100, range=(-0.7,0.7))
bin_centers_res = (bin_edges_res[1:] + bin_edges_res[:-1])/2
s_counts_res = np.sqrt(counts_res)

x = bin_centers_res[counts_res>0]

y = counts_res[counts_res>0]

sy = s_counts_res[counts_res>0]

hist_res = ax.hist(data, bins = 100, histtype="step", label = "Distribution of residuals")

#def cauchy (x,mu,gamma):
#    1./(np.pi*gamma*(1+((x-mu)/gamma)**2))

function_better_vec = np.vectorize(cauchy)

x = np.asarray(x)
y = np.asarray(y)
sy = np.asarray(sy)

print (len(x))
chi2_res = Chi2Regression(gauss_extended, x, y, sy)
minuit_res = Minuit(chi2_res, N1 = 1.0, mu1 = 0.001, sigma1 = 0.06, N2 = 10, mu2= -0.01, sigma2 = 0.15 ,pedantic = False)
minuit_res.migrad()
xaxis = np.linspace(-1.0, 1.0, 1000)
yaxis = gauss_extended(xaxis, *minuit_res.args)
ax.plot(xaxis, yaxis, '-', label='Fit to double gaussian')
ndof_res = len(x)-len(minuit_res.args)
print (minuit_res.fval, (stats.chi2.sf(minuit_res.fval,ndof_res))*100, ndof_res)

names = ['Entire period:','Entries','Mean','RMS', "Chi2/Ndof", "Prob"]
values = [
          "",
          "{:d}".format(len(x)),
          "{:.3f}".format(x.mean()),
          "{:.3f}".format(x.std()),
          "{0:.3f}/{1:.0f}".format(minuit_res.fval, ndof_res),
          "{0:.3f}".format(stats.chi2.sf(minuit_res.fval, ndof_res))
          ]
ax.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax.transAxes, fontsize=10, verticalalignment='top')
ax.set_xlabel("Time")
ax.set_ylabel("Frequency")
#ax1.plot(xaxis, yaxis, '-', label='Fit to gaussian')
ax.legend(loc = "best")
if SavePlots:
    fig.savefig("dobblegauss.png")


fig2, ax2 = plt.subplots (figsize = (10,5))

hist_use = ax2.hist(data, bins = 100, histtype="step", label = "Distribution of residuals")

print (len(x))
chi2_res_1 = Chi2Regression(gauss_extended_norm, x, y, sy)
minuit_res_1 = Minuit(chi2_res_1, N = 1.0, mu = 0.001, sigma = 0.06, pedantic = False)
minuit_res_1.migrad()
xaxis = np.linspace(-1.0, 1.0, 1000)
yaxis = gauss_extended_norm(xaxis, *minuit_res_1.args)
ax2.plot(xaxis, yaxis, '-', label='Fit to gaussian')
ndof_res_1 = len(x)-len(minuit_res_1.args)
print (minuit_res_1.fval, (stats.chi2.sf(minuit_res_1.fval,ndof_res_1))*100, ndof_res_1)
names = ['Entire period:','Entries','Mean','RMS', "Chi2/Ndof", "Prob"]
values = [
          "",
          "{:d}".format(len(x)),
          "{:.3f}".format(x.mean()),
          "{:.3f}".format(x.std()),
          "{0:.3f}/{1:.0f}".format(minuit_res_1.fval, ndof_res_1),
          "{0:.3f}".format(stats.chi2.sf(minuit_res_1.fval, ndof_res_1))
          ]
ax2.text(0.05, 0.95, nice_string_output(names, values), family='monospace', transform=ax2.transAxes, fontsize=10, verticalalignment='top')
ax2.set_xlabel("Time")
ax2.set_ylabel("Frequency")
#ax1.plot(xaxis, yaxis, '-', label='Fit to gaussian')
ax2.legend(loc = "best")

if SavePlots:
    fig2.savefig("Gauss.png")










plt.show(block=False)
try:
    __IPYTHON__
except:
    raw_input('Press Enter to exit')
