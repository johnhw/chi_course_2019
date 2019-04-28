import pykalman
import numpy as np
import tkanvas

# Draw each step of the Kalman filter onto a TKinter canvas
def draw_kalman_filter(src): 
    src.clear()
    # update paths and draw them
    for p in src.obs_path:
        src.circle(p[0], p[1], 2, fill="white")
    for p in src.track_path:
        src.circle(p[0], p[1], 2, fill="blue")
    track = not np.any(np.isnan(src.obs[0]))
       
    if track:
        src.obs_path.append(src.obs)
        
    src.track_path.append(src.new_mean[:2])
    
    # draw the prior
    src.normal(src.mean[:2], src.cov[:2,:2], outline="blue")   
    text = src.text(10,10,text="Prior",anchor="w", fill="gray")
    yield 0  # this is a trick to allow to "return" here but resume later
    ax = np.dot(src.A, src.mean)    
    acov = np.dot(np.dot(src.A, src.cov), src.A.T)        
    # prediction after linear dynamics
    src.normal(ax[:2], acov[:2,:2], outline="green",  dash=(2, 4))
    src.modify(text, text="Prediction")
    yield 0
    if track:
        # observation (if there is one)
        src.circle(src.obs[0], src.obs[1], 5, fill="white")
        src.modify(text, text="Observation")
        yield 0
        # uncertainty of observation
        src.normal(src.obs, src.sigma_c[:2,:2], outline="purple", dash=(2,2))        
        src.modify(text, text="Observation uncertainty")
        yield 0                     
    # posterior
    src.normal(src.new_mean[:2], src.new_cov[:2,:2], outline="lightblue")
    src.modify(text, text="Posterior")
    yield 0
    
    
   
# draw the Kalman filter updates interactively
def kalman_draw(src):
    if src.frame_time>10:
        # slowly speed up over time
        src.frame_time = src.frame_time*0.93
    try:
        src.kalman_iter.next()
    # we've drawn all the steps, so make another update
    except StopIteration:
        src.mean, src.cov = src.new_mean, src.new_cov
        src.obs = src.path[src.time_step]
        src.time_step += 1
        
        # for some range of values, disable observations
        if not np.any(np.isnan(src.obs)):
            src.new_mean, src.new_cov = src.kalman_filter.filter_update(src.mean, src.cov, observation=src.obs)
        else:
            # no observation available
            src.new_mean, src.new_cov = src.kalman_filter.filter_update(src.mean, src.cov)
        
        if src.time_step>=len(src.path):
            src.quit(None)
        else:
            src.kalman_iter = draw_kalman_filter(src)    
                
                
def run_kalman(path, mu_0, sigma_0, A, C, sigma_a, sigma_c, frame_time=2000):
   
    kalman_filter = pykalman.KalmanFilter(
    transition_matrices = A,
    observation_matrices = C,
    transition_covariance = sigma_a,
    observation_covariance = sigma_c,
    initial_state_mean = mu_0,
    initial_state_covariance = sigma_0
    )
    
    src = tkanvas.TKanvas(draw_fn=kalman_draw, frame_time=frame_time)   
    mean, cov = mu_0, sigma_0

    obs = path[0]
    # initial update
    new_mean, new_cov = kalman_filter.filter_update(mean, cov, observation=obs)
    src.mean = mean
    src.cov = cov
    src.new_mean = new_mean
    src.new_cov = new_cov
    src.A = A
    src.C = C
    src.sigma_a = sigma_a
    src.sigma_c = sigma_c
    src.kalman_filter = kalman_filter
    src.track = True
    src.obs_path = []
    src.track_path = [] 
    src.obs = obs
    src.path = path
    src.time_step = 1
    src.kalman_iter = draw_kalman_filter(src)  
   