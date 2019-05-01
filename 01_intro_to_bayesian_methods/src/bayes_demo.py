# numpy, scipy
import numpy as np
import scipy.stats
import scipy
from scipy.stats import norm
import numpy.ma as ma
import time
import IPython
import matplotlib as mpl
import matplotlib.pyplot as plt

def redraw_figure(fig):
    IPython.display.clear_output(wait=True)
    IPython.display.display(fig)
    
def prior_posterior(prior_mean=0, prior_std=1, ev_std=0.5, n=10, anim=False):
    # initial configurateion
    mean = prior_mean
    std = prior_std
    var = std*std
    prior = scipy.stats.norm(mean,std)
    evidence = scipy.stats.norm(1, 0.25)
    sample_var = ev_std**2 # the *expected* variance of our observations
    
    #########################
    # create a figure to plot on
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    xs = np.linspace(-5,5,200)
    ax.plot(xs, prior.pdf(xs), label="Prior belief", alpha=1)
    ax.plot(xs, evidence.pdf(xs), label="True generating PDF", alpha=1)        
    
    # note that changing this allows us to continously adjust our belief
    # in our observations 
    ax.plot([0,0],[0,-0.1], 'c', alpha=0.7, label="Evidence")
    ax.plot([0,0],[0,-0.1], 'k:', alpha=0.7, label="Posterior belief")
    ax.set_title("Recursive Bayesian estimation")
    
    # label etc.
    ax.set_xlabel("x")
    ax.set_ylabel("PDF $f_X(x)$")    
    ax.legend()
    ax.set_frame_on(False)
    ax.set_ylim(-1.8, 3.0)
    #
    
    ##################################
    # plot each sample and updated belief
    for i in range(n):
        
        # draw a sample from the process
        sample = evidence.rvs()
        
        old_mean, old_var = mean, var
        # single step update for a normal distribution    
        # this is essentially a 1D Kalman filter        
        mean = (var * sample + sample_var * mean) / (sample_var + var)
        var = (var*sample_var) / (sample_var+var)     
        
        # create the PDF of the observation from our process
        sample_pdf = scipy.stats.norm(sample, ev_std).pdf
        
       
        # draw the curves
        ax.fill_between(xs, scipy.stats.norm(old_mean, np.sqrt(old_var)).pdf(xs), alpha=0.1, color='C2')
        
        if anim:            
            time.sleep(0.5)
            redraw_figure(fig)
            
        # plot the sample and the resulting pdf
        ax.plot([sample,sample],[0,-0.5], 'c', alpha=0.7)
        if anim:
            ax.plot(xs,-sample_pdf(xs), 'c', alpha=0.25)        
            time.sleep(0.25)
            redraw_figure(fig)
        #ax.set_ylim(-1, 2.0)
        
        
    ax.legend()