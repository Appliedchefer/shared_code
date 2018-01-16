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
def mu_RMS (input_data = []):

    N = len(input_data)

    mu = np.sum(input_data)/N

    RMS = np.sqrt(np.sum((input_data-mu)**2)/(N-1))

    errmu = RMS/np.sqrt(N)

    return mu, RMS, errmu

def nice_string_output(names, values, extra_spacing = 2):
    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values):
        string += "{0:s} {1:>{spacing}} \n".format(name, value,
                   spacing = extra_spacing + max_values + max_names - len(name))
    return string[:-2]
data = []
A_list = []
B_list = []
C_list = []
ill_list = []
with open( 'data_FisherSyndrome.txt', 'r' ) as infile :
    counter = 0

    for line in infile:
        line = line.strip().split()
        data.append([float(line[2]), float(line[3]), float(line[4]), int(line[0])])

        isIll = data[-1][3]
        A = data[-1][0]
        B = data[-1][1]
        C = data[-1][2]


        A_list.append(A)
        B_list.append(B)
        C_list.append(C)
        ill_list.append(isIll)
        # Print some numbers as a sanity check:
        if (counter < 10) :
            print("  Reading data:   Is the person ill? {0:d}   A = {1:5.2f}   B = {2:5.2f}   C = {3:5.2f}".format(isIll, A, B, C))
        counter += 1

data = np.array(data)
A_list = np.array(A_list)
B_list = np.array(B_list)
C_list = np.array(C_list)
ill_list = np.array(ill_list)

healthy_list = []
for i in range(len(ill_list)):
    if ill_list[i] == 0:
        healthy_list.append(ill_list[i])


healthy_list = np.array(healthy_list)

A_ill = []
B_ill = []
C_ill = []
for i in range (2000):
    A_ill.append(A_list[i])
    B_ill.append(B_list[i])
    C_ill.append(C_list[i])

A_ill = np.array(A_ill)
B_ill = np.array(B_ill)
C_ill = np.array(C_ill)
A_healthy = []
B_healthy = []
C_healthy = []

for i in range(len(ill_list)):
    if ill_list[i] == 0:
        A_healthy.append(A_list[i])
        B_healthy.append(B_list[i])
        C_healthy.append(C_list[i])

A_healthy = np.array(A_healthy)
B_healthy = np.array(B_healthy)
C_healthy = np.array(C_healthy)

#4.1
counts_ill, bin_edges_ill = np.histogram(A_ill, bins=120, range=(0, 30))
bin_centers_ill = (bin_edges_ill[1:] + bin_edges_ill[:-1])/2
s_counts_ill = np.sqrt(counts_ill)

x = bin_centers_ill[counts_ill>0]

y = counts_ill[counts_ill>0]

sy = s_counts_ill[counts_ill>0]

fig2, ax2 = plt.subplots (figsize = (10,5))
ax2.errorbar (x, y, yerr=sy, label='Distribution of A ill',
            fmt='.k',  ecolor='k', elinewidth=1, capsize=1, capthick=1)

A_ill_mean, A_ill_sigma, A_ill_error_mean = mu_RMS(A_ill)

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def gauss_extended(x, N, mu, sigma) :
    """Non-normalized Gaussian"""
    return N * gauss_pdf(x, mu, sigma)




chi_A_ill = Chi2Regression(gauss_extended, x, y, sy)
minuit_ill = Minuit(chi_A_ill, pedantic=False, N=len(A_ill), mu = A_ill_mean, sigma = A_ill_sigma)
minuit_ill.migrad()
print(minuit_ill.fval)
#Ndof = 120-*minuit_ill.args
prob = stats.chi2.sf(minuit_ill.fval, (65-len(minuit_ill.args)))
print (prob)

xaxis = np.linspace(0.0, 30, 1000)
yaxis = gauss_extended(xaxis, *minuit_ill.args)
names = ['Distribution of A ill:','Entries','Mean','RMS', "Chi2/Ndof", "Prob"]
values = ["",
          "{:d}".format(len(A_ill)),
          "{:.3f}".format(x.mean()),
          "{:.3f}".format(x.std(ddof=1)),
          "{0:.3f}/{1:.0f}".format(minuit_ill.fval, len(x)-len(minuit_ill.args)),
          "{0:.3f}".format(stats.chi2.sf(minuit_ill.fval, (len(x)-len(minuit_ill.args))))
          ]
ax2.text(0.005, 0.85, nice_string_output(names, values), family='monospace', transform=ax2.transAxes, fontsize=10, verticalalignment='top')
ax2.set_xlabel("Distribution of A")
ax2.set_ylabel("Frequency")
ax2.plot(xaxis, yaxis, '-', label='Fit to gaussian')
ax2.legend(loc = "best")
if SavePlots:
    fig2.savefig("A_ill.png")

#4.2
check = stats.pearsonr(B_ill, C_ill)
print (check)

#4.3



counts_ill_C, bin_edges_ill = np.histogram(C_ill, bins=200, range=(-4, 3))
bin_centers_ill = (bin_edges_ill[1:] + bin_edges_ill[:-1])/2
sum_c_ill = np.sum(counts_ill_C)
print (sum_c_ill)

counts_healthy_C, bin_edges_healthy = np.histogram(C_healthy, bins=200, range=(-4, 3))
bin_centers_healthy = (bin_edges_healthy[1:] + bin_edges_healthy[:-1])/2
sum_c_healthy = np.sum(counts_healthy_C)

fig3, ax3 = plt.subplots (figsize = (10,5))
test_A = ax3.hist(C_ill, bins=200, range=(-4, 4), histtype='step', linewidth=2, label='C people ill', color='red')
test_B = ax3.hist(C_healthy, bins=200, range=(-4, 4), histtype='step', linewidth=2, label='C people healthy', color='blue')
ax3.set_xlabel("Distribution of C people")
ax3.set_ylabel("Frequency")
ax3.legend(loc="best")

if SavePlots:
    fig3.savefig("C_dist.png")

print (counts_ill_C)
accsum_c_ill = 0
for i in range (len (counts_ill_C)):
    accsum_c_ill += counts_ill_C[i]


    proc = accsum_c_ill/sum_c_ill



    if proc > 0.99:
        this_i = i
        break

print ("My alpha for C_ill",this_i, proc)

accsum_c_healthy = 0
for i in range (this_i):
        accsum_c_healthy += counts_healthy_C[i]


proc2 = accsum_c_healthy/sum_c_healthy
print ("What is declares sick, when healthy C (beta)",proc2)


fig4, ax4 = plt.subplots(figsize = (10,5))


F_ill = -1.33*A_ill + 0.63*B_ill + 2.1*C_ill
F_healthy = -1.33*A_healthy + 0.63*B_healthy + 2.1*C_healthy

counts_F_ill , counts_F_ill_edges = np.histogram(F_ill, bins=200, range=(0, 30))
Fsum_ill = np.sum (counts_F_ill)

counts_F_healthy , counts_F_healthy_edges = np.histogram(F_healthy, bins=200, range=(0, 30))
Fsum_healthy = np.sum (counts_F_healthy)

F_ill_hist = ax4.hist(F_ill ,bins=200, range=(0, 30), histtype='step', linewidth=2, label='Distribution of F ill', color='red')
F_healthy_hist = ax4.hist(F_healthy ,bins=200, range=(0, 30), histtype='step', linewidth=2, label='Distribution of F healthy', color='blue')
ax4.set_xlabel("Distribution of F people")
ax4.set_ylabel("Frequency")
ax4.legend(loc="best")
if SavePlots:
    fig4.savefig("F_dist.png")
accum_F_ill = 0.0
for i in range (len (counts_F_ill)):
    accum_F_ill += counts_F_ill[i]

    procF_i = accum_F_ill/Fsum_ill

    if procF_i > 0.99:
        si = i
        break

print ("My alpha for F ill",si, procF_i)

accum_F_healthy = 0.0
for i in range ( si ):
    accum_F_healthy += counts_F_healthy[i]


procF_h = accum_F_healthy /Fsum_healthy

print ("My beta for F_healthy",procF_h)













plt.show(block=False)
try:
    __IPYTHON__
except:
    raw_input('Press Enter to exit')
