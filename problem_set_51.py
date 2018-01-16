from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from probfit import BinnedLH, Chi2Regression, Extended
from scipy.special import erfc
from scipy import stats


SavePlots = True


#Functions
def nice_string_output(names, values, extra_spacing = 2):
    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values):
        string += "{0:s} {1:>{spacing}} \n".format(name, value,
                   spacing = extra_spacing + max_values + max_names - len(name))
    return string[:-2]

def fit_func_cons (x, p0):
    return p0 + 0*x

def fit_func_12 (x, p0, p1):
    return p0 + p1*x

def fit_func_12_more (x, p0, p1):
    return p0 + p1*x

def fit_func_all(x, p0, p1, p2, p3, p4, p5):
    if x < 32:
        return (p0 * 1./(1.+np.exp(-(x-p2)*p3)) + p1)
    else:
        return p4 + p5*x
# ----------------------------------------------------------------------------------- #
# Read data:
# ----------------------------------------------------------------------------------- #

# Define input data lists:
x   = []
ex  = []
y   = []
ey  = []

with open( 'data_LukeLightningLights.txt', 'r' ) as infile :
    for line in infile:
        line = line.strip().split()
        x.append(float(line[0]))
        ex.append(float(line[1]))
        y.append(float(line[2]))
        ey.append(float(line[3]))

        # Print the numbers as a sanity check (only 41 in total):
        print("  Read data:  {0:6.3f}  {1:6.3f}  {2:6.3f}  {3:6.3f}".format(x[-1], ex[-1], y[-1], ey[-1]))


x = np.array(x)
ex = np.array (ex)
y = np.array (y)
ey = np.array(ey)

fig1, ax1 = plt.subplots (figsize = (10,5))

x_12 = np.array(x [0:12])
y_12 = np.array(y [0:12])
ey_12 = np.array(ey [0:12])

ax1.errorbar (x_12, y_12, yerr=ey_12, label='Monthly income',
            fmt='bo',  ecolor='b', elinewidth=1, capsize=1, capthick=1)

chi2_cons = Chi2Regression(fit_func_cons, x_12, y_12, ey_12)
minuit_cons = Minuit(chi2_cons, pedantic = False, p0 = -0.4,  print_level = 0)
minuit_cons.migrad()

ax1.plot(x_12, fit_func_cons(x_12, *minuit_cons.args), "-r", label ="Fit to a constant line")
ndof_cons = len(x_12)- len(minuit_cons.args)
print (ndof_cons)
print (minuit_cons.fval, (stats.chi2.sf(minuit_cons.fval,ndof_cons)*100))
names = ['Check if constant:','Entries','Mean','RMS', "Chi2/Ndof", "Prob"]
values = ["",
          "{:d}".format(len(y_12)),
          "{:.3f}".format(y_12.mean()),
          "{:.3f}".format(y_12.std(ddof=1)),
          "{0:.3f}/{1:.0f}".format(minuit_cons.fval, ndof_cons),
          "{0:.3f}".format(stats.chi2.sf(minuit_cons.fval, ndof_cons))
          ]
ax1.text(0.05, 0.85, nice_string_output(names, values), family='monospace', transform=ax1.transAxes, fontsize=10, verticalalignment='top')
ax1.set_xlabel("Months")
ax1.set_ylabel("Income in M$")
#ax1.plot(xaxis, yaxis, '-', label='Fit to gaussian')
ax1.legend(loc = "best")
if SavePlots:
    fig1.savefig("12_months.png")
fig2, ax2 = plt.subplots (figsize = (10,5))


chi2_12 = Chi2Regression(fit_func_12, x_12, y_12, ey_12)
minuit_12 = Minuit(chi2_12, pedantic = False, p0 = -0.4, p1 = 1, print_level = 0)
minuit_12.migrad()
ax2.errorbar(x_12, y_12, yerr=ey_12, label='Monthly income',
            fmt='.k',  ecolor='k', elinewidth=1, capsize=1, capthick=1)
ax2.plot(x_12,fit_func_12(x_12,*minuit_12.args), label = "Linear fit")
print (minuit_12.errors)

ndof_12 = len(x_12)-len(minuit_12.args)
print (minuit_12.fval, (stats.chi2.sf(minuit_12.fval,ndof_12)*100))
names = ['Linear fit:','Entries','Mean','RMS', "Chi2/Ndof", "Prob"]
values = ["",
          "{:d}".format(len(x_12)),
          "{:.3f}".format(y_12.mean()),
          "{:.3f}".format(ey_12.std()),
          "{0:.3f}/{1:.0f}".format(minuit_12.fval, ndof_12),
          "{0:.3f}".format(stats.chi2.sf(minuit_12.fval, ndof_12))
          ]
ax2.text(0.05, 0.85, nice_string_output(names, values), family='monospace', transform=ax1.transAxes, fontsize=10, verticalalignment='top')
ax2.set_xlabel("Months")
ax2.set_ylabel("Income in M$")
#ax1.plot(xaxis, yaxis, '-', label='Fit to gaussian')
ax2.legend(loc = "best")
if SavePlots:
    fig2.savefig("12_months_linear.png")

fig3, ax3 = plt.subplots (figsize = (10,5))

chi2_12_more = Chi2Regression(fit_func_12_more, x[0:15], y[0:15], ey[0:15])
minuit_12_more = Minuit(chi2_12_more, pedantic = False, p0 = -0.4, p1 = 1, print_level = 0)
minuit_12_more.migrad()
ax3.errorbar(x[0:15], y[0:15], yerr=ey[0:15],label='Monthly income',
            fmt='bo',  ecolor='b', elinewidth=1, capsize=1, capthick=1)
ax3.plot(x[0:15],fit_func_12_more(x[0:15],*minuit_12_more.args), label = "Fit to linear")

ndof_12_more = len(x[0:15])-len(minuit_12_more.args)
print (minuit_12_more.fval, (stats.chi2.sf(minuit_12_more.fval,ndof_12_more)*100))

names = ['Linear fit:','Entries','Mean','RMS', "Chi2/Ndof", "Prob"]
values = ["",
          "{:d}".format(len(x[0:15])),
          "{:.3f}".format(y[0:15].mean()),
          "{:.3f}".format(y[0:15].std()),
          "{0:.3f}/{1:.0f}".format(minuit_12_more.fval, ndof_12_more),
          "{0:.3f}".format(stats.chi2.sf(minuit_12_more.fval, ndof_12_more))
          ]
ax3.text(0.05, 0.85, nice_string_output(names, values), family='monospace', transform=ax1.transAxes, fontsize=10, verticalalignment='top')
ax3.set_xlabel("Months")
ax3.set_ylabel("Income in M$")
#ax1.plot(xaxis, yaxis, '-', label='Fit to gaussian')
ax3.legend(loc = "best")
if SavePlots:
    fig3.savefig("15_months.png")

"""
chi2_12_more = Chi2Regression(fit_func_12_more, x, y, ey)
minuit_12_more = Minuit(chi2_12_more, pedantic = False, p0 = -0.4, p1 = 1, print_level = 0)
minuit_12_more.migrad()
ax3.errorbar(x, y, yerr=ey)
ax3.plot(x,fit_func_12_more(x,*minuit_12_more.args))

ndof_12_more = len(x)-len(minuit_12_more.args)
print (minuit_12_more.fval, (stats.chi2.sf(minuit_12_more.fval,ndof_12_more)*100))
"""

fig4, ax4 = plt.subplots(figsize =(10,5))
chi2_32 = Chi2Regression(fit_func_12_more, x[25:31], y[25:31], ey[25:31])
minuit_32 = Minuit(chi2_32, pedantic = False, p0 = 2, p1 = 0.05, print_level=0)
minuit_32.migrad()
ax4.errorbar(x[25:31], y[25:31], ey[25:31], label='Monthly income',
            fmt='bo',  ecolor='b', elinewidth=1, capsize=1, capthick=1)
ax4.plot(x[25:31], fit_func_12_more(x[25:31], *minuit_32.args), label = "Fit to linear")
ndof_32 = len(x[25:31])-len(minuit_32.args)
print (minuit_32.fval, (stats.chi2.sf(minuit_32.fval,ndof_32)*100))
print (minuit_32.args, minuit_32.errors)
p0, p1 = minuit_32.args
print (p1)
ep0, ep1 = minuit_32.errors
print (ep1)

names = ['To estimate for month 32:','Entries','Mean','RMS', "Slope", "Chi2/Ndof", "Prob"]
values = ["",
          "{:d}".format(len(y[25:31])),
          "{:.3f}".format(y[25:31].mean()),
          "{:.3f}".format(y[25:31].std()),
          "{0:.3f} +- {1:.3f}".format(p1, 0.02629502939695673),
          "{0:.3f}/{1:.0f}".format(minuit_32.fval, ndof_32),
          "{0:.3f}".format(stats.chi2.sf(minuit_32.fval, ndof_32))
          ]
ax4.text(0.05, 0.85, nice_string_output(names, values), family='monospace', transform=ax1.transAxes, fontsize=10, verticalalignment='top')
ax4.set_xlabel("Months")
ax4.set_ylabel("Income in M$")
#ax1.plot(xaxis, yaxis, '-', label='Fit to gaussian')
ax4.legend(loc = "best")

if SavePlots:
    fig4.savefig("diff_32.png")

diff = p0+32*p1-y[31]
print (diff)
error_diff = ((0.750752637991111)**2+(0.02629502939695673)**2)**(0.5)
print (error_diff)


fig5, ax5 = plt.subplots(figsize =(10,5))
def func_all(x, p0, p1, p2, p3, p4, p5) :
    if x < 32:
        return (p0 * 1./(1+np.exp(-(x-p2)*p3)) + p1)
    else:
        return p4 + p5*x
func_advanced_vec = np.vectorize(func_all)
# Above is a function that transforms a non-vectorized function to vectorized,
# such that func_advanced_vec allows x to be a numpy array instead of just a scalar
x = np.asarray(x)
y = np.asarray(y)
ey = np.asarray(ey)

chi2_all = Chi2Regression(func_all, x, y, ey)

minuit_all = Minuit(chi2_all, pedantic=False,
                         p0 = 4, limit_p0=(4, 5), p1 = 1, p2=20, limit_p2=(20, 1000), p3=5.0,
                         p4=0.5, p5=1)
minuit_all.migrad() # fit
p0, p1, p2, p3, p4, p5 = minuit_all.args
print("Advanced fit")
for name in minuit_all.parameters:
    print("Fit value: {0} = {1:.5f} +/- {2:.5f}".format(name, minuit_all.values[name], minuit_all.errors[name]))

x_fit = np.linspace(1, 41, 1000)
y_fit_all = func_advanced_vec(x_fit, p0, p1, p2, p3, p4, p5)
ax5.errorbar (x, y, yerr = ey, label='Monthly income',
            fmt='bo',  ecolor='b', elinewidth=1, capsize=1, capthick=1)
ax5.plot(x_fit, y_fit_all, '-', label = "Fit to entire period")
ndof_all = len(x)-len(minuit_all.args)
print (minuit_all.fval, (stats.chi2.sf(minuit_all.fval,ndof_all))*100)

names = ['Sigmoid fit:','Linear fit:','Entire period:','Entries','Mean','RMS', "Chi2/Ndof", "Prob"]
values = ["p0*1./(1+np.exp(-(x-p2)*p3))+p1",
          "p4 + p5*x",
          "",
          "{:d}".format(len(y)),
          "{:.3f}".format(y.mean()),
          "{:.3f}".format(y.std()),
          "{0:.3f}/{1:.0f}".format(minuit_all.fval, ndof_all),
          "{0:.3f}".format(stats.chi2.sf(minuit_all.fval, ndof_all))
          ]
ax5.text(0.45, 0.4, nice_string_output(names, values), family='monospace', transform=ax1.transAxes, fontsize=10, verticalalignment='top')
ax5.set_xlabel("Months")
ax5.set_ylabel("Income in M$")
#ax1.plot(xaxis, yaxis, '-', label='Fit to gaussian')
ax5.legend(loc = "best")

if SavePlots:
    fig5.savefig("sigmoid.png")

plt.show(block=False)
try:
    __IPYTHON__
except:
    raw_input('Press Enter to exit')
