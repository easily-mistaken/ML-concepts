from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
# Useful module for dealing with the Gaussian density
from scipy.stats import norm, multivariate_normal
# installing packages for interactive graphs
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider

data = np.loadtxt('GH3.txt', delimiter=',')
# Names of features
featurenames = ['Floor', 'Weight']

np.random.seed(0)
perm = np.random.permutation(138)
trainx = data[perm[0:138],1]
trainy = data[perm[0:138],0]

def fit_generative_model(x,y):
    k = 4 # number of classes
    mu = np.zeros(k+1) # list of means
    var = np.zeros(k+1) # list of variances
    pi = np.zeros(k+1) # list of class weights
    for label in range(0,k):
        indices = (y==label)
        mu[label] = np.mean(x[indices])
        var[label] = np.var(x[indices])
        pi[label] = float(sum(indices))/float(len(y))
    return mu, var, pi

@interact_manual( )
def show_densities():
    mu, var, pi = fit_generative_model(trainx, trainy)
    colors = ['r', 'k', 'g','b']
    for label in range(0,4):
        m = mu[label]
        s = np.sqrt(var[label])
        x_axis = np.linspace(m - 3*s, m+3*s, 1000)
        plt.plot(x_axis, norm.pdf(x_axis,m,s), colors[label], label="class " + str(label))
    plt.xlabel(featurenames[0], fontsize=14, color='red')
    plt.ylabel('Density', fontsize=14, color='red')
    plt.legend()
    plt.show()

wt = input('Enter weight of the candidate: ')

mx = 0
floor = 1
score = np.zeros((4))
for label in range(4):
    score[label] = norm.pdf(wt,mu[label],np.sqrt(var[label]))*pi[label]
    if(mx<score[label]):
        mx = score[label]
        floor = label

print('We predict the floor to be',floor)
