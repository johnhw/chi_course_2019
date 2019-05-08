import numpy as np
import pfilter


def prior(n):
    """Return n initial draws from the prior over the position
    and velocity of the cursor before any observations have been drawn"""
    x_prior = np.random.uniform(0, 1, n)  # anywhere 0->1
    dx_prior = np.random.normal(0, 0.2, n)  # slow movement
    return np.stack([x_prior, dx_prior]).T


def dynamics(particles, dt):
    """Apply our very simple dynamics, with velocity and some
    random noise, and return a new set of particles"""

    new_particles = np.array(particles)  # copy
    new_particles[:, 0] += particles[:, 1] * dt  # integrate
    # diffuse
    new_particles += np.random.normal(0, 1, particles.shape) * [1e-2, 1e-1]
    return new_particles


def observation(particles):
    """Project from 
    internal state (x, dx) => observed states (x, speed)"""

    x = particles[:, 0]
    speed = np.abs(particles[:, 1])
    # observations
    return np.stack([x, speed]).T


import numpy.ma as ma


def weighting(hypothesised, real):
    """Compare a set of hypothesised observation values (one) real observation
     and return a unnormalised weighting for each particle"""

    # position, speed weights
    # (note: these can be masked and therefore not contribute to the calculation)
    weights = [50.0, 10.0]

    # squared difference, weighted and exponentiated
    # this gives a similarity measure
    difference = np.nansum((hypothesised - real) ** 2 * weights, axis=1)
    weight = np.exp(-difference)
    
    weight[hypothesised[:,0]<0] = 0.0
    weight[hypothesised[:,0]>1] = 0.0
    return weight + 1e-6


def filter_step(particles, observed, dt=0.01, prior_rate=0.05):
    """Update one complete step given a set of particles
    and an observation.
    
        Steps:
        * Apply dynamics to the particles
        * Compare with observations to get weights
        * Normalise weights
        * Resample particles according to weights
        * Replace a small fraction of particles with 
            prior draws to "refresh" the sampler

    """

    new_particles = dynamics(particles, dt)  # dynamics

    # replace a few particles with draws from the posterior
    prior_draws = np.random.uniform(0, 1, len(particles)) < prior_rate
    new_particles[prior_draws] = prior(np.sum(prior_draws))

    weights = weighting(observation(new_particles), observed)  # weighting
    normalised_weights = weights / np.sum(weights)  # normalise weights

    n_eff = (1.0 / np.sum(normalised_weights ** 2)) / len(particles)
    
    

    new_particles = new_particles[pfilter.resample(normalised_weights)]  # resampling

    return new_particles, normalised_weights


def expected_position(particles, normalised_weights):
    """Return the expectation of the particle position/speed"""
    return np.sum((particles.T * normalised_weights.T).T, axis=0)


def variance_position(particles, weights):
    ex_pos = expected_position(particles, weights)
    sqr_dif = (ex_pos - particles) ** 2
    return np.sum((sqr_dif.T * weights.T).T, axis=0)

