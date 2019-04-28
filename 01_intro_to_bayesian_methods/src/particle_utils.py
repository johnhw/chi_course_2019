import numpy as np
import tkanvas
import matplotlib

import matplotlib.pyplot as plt


def plot_pfilter(time,expected, observed, particles, weights, means):
    """Apply a particle filter to a time series, and plot the
    first component of the predictions alongside the expected
    output."""
    # expected output
    plt.plot(time, expected, 'C1', lw=3)
    plt.plot(time, observed, '+C3', lw=3)
    
    # particles 
    ts = np.tile(time[:,None], particles.shape[1]).ravel()
    weights =  weights.ravel()    
    rgba_colors = np.zeros((len(weights),4))
    rgba_colors[:,0:3] = matplotlib.colors.to_rgb('C2')
    weights *= 10
    rgba_colors[:, 3] = np.clip(weights, 0, 1)
    plt.scatter(ts, particles[:,:,0].ravel(),  c=rgba_colors, s=weights*200)
    # mean estimation
    plt.plot(time, means, 'C0--', lw=2)
    # legend
    plt.legend(["True", "Observed", "Mean estimate", "Particle"])
    plt.xlabel("Time")
    plt.ylabel("X")
    plt.title("Particle filter estimate")
    
    
frame = 0

def animate_pfilter(ts, xs, ys, particles, weights, means, yscale=1.0):
    global frame
    w,h = 1200,600
    
    yoff = h/2
    xoff = 0
    yscale = yscale * h/3
    xscale = float(w) / np.max(ts)
    frame = 0
    def draw_pf(src):
        global frame
        i = frame
        t,x,y = ts[i], xs[i], ys[i]
        particle_set = particles[i]
        weight = weights[i]
        mean = means[i]
        tdraw = t[0]*xscale+xoff
        
        for w, particle in zip(weight, particle_set):        
            src.circle(tdraw, particle[0]*yscale+yoff, 1, fill="white", outline='')
        
        max_p = np.argmax(weight)        
        src.circle(tdraw, particle_set[max_p][0]*yscale+yoff, 5, fill="green")
        
        if not np.isnan(y):
            src.circle(tdraw, y[0]*yscale+yoff, 5, fill="orange")
        src.circle(tdraw, x[0]*yscale+yoff, 4, fill="red")
        src.circle(tdraw, mean[0]*yscale+yoff, 5, fill="cyan")
        frame += 1

    kanvas = tkanvas.TKanvas(draw_fn=draw_pf, w=w, h=h)
    
    