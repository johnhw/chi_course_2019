import tkanvas
tkanvas = reload(tkanvas)
from tkanvas import TKanvas
import numpy as np

class Gilbert(object):
    """implementS a Gilbert Markov chain, which can switch between two states. Simulates
    bursty behaviour."""    
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.state = 0        
    
    def update(self, dt):
        # account for sampling rate              
        p1 = 1-((1-self.p1)**(dt))
        p2 = 1-((1-self.p2)**(dt))
        
        if self.state==0:            
            if np.random.random()<p1:
                self.state = 1
        if self.state==1:            
            if np.random.random()<p2:
                self.state = 0
        return self.state
                
class NoiseCursor(object):
    def __init__(self, filter=None, noise=0, dropout=[0, 0.1], jump=0.0, jump_scale=200):
        self.noise = noise
        self.dropout = dropout
        self.filter = filter
        self.jump_scale = jump_scale
        self.jump = jump
        self.gilbert = Gilbert(dropout[0], dropout[1])
        self.true_history = []
        self.raw_history = []
        self.filtered_history = []
        
        
    def update(self, x, y):
        true_x, true_y = x,y
        x = x + np.random.normal(0,1)*self.noise
        y = y + np.random.normal(0,1)*self.noise
        if np.random.uniform(0,1)<self.jump:
            y += np.random.uniform(-self.jump_scale, self.jump_scale)
            x += np.random.uniform(-self.jump_scale, self.jump_scale)
        cutout = self.gilbert.update(1)
        if cutout:
            x,y = np.nan, np.nan
        ret = None
        
        if self.filter:
            ret = self.filter(x,y)               
            if type(ret)==type({}):
                fx, fy = ret["mean"]
            else:
                fx, fy = ret
                ret = None
        else:
            fx,fy = x, y
            
        self.true_history.append((true_x, true_y))
        self.filtered_history.append((fx, fy))
        self.raw_history.append((x, y))
        
        return x,y, fx, fy, true_x, true_y, ret
    
    
def in_rect(pt, rect):
    x,y = pt
    x1,y1,x2,y2 = rect
    return (x>x1 and x<x2 and y>y1 and y<y2)
        
import time
    
class NoiseCursorDemo(object):
    def __init__(self, **kwargs):
        self.cursor = NoiseCursor(**kwargs)
        self.screen_size = 800
        
        self.target_size = 20
        
        self.hits = 0
        self.misses = 0
        self.last_filter = (0,0)
        self.start_time = time.time()
        self.new_target()
        c = TKanvas(draw_fn=self.draw, event_fn=self.event, quit_fn=self.quit, w=self.screen_size, h=self.screen_size)     
        
    def get_traces(self):
        return {"true":self.cursor.true_history,
            "raw":self.cursor.raw_history,
            "filtered":self.cursor.filtered_history}
            
        
    def new_target(self):    
        x, y = np.random.uniform(0+self.target_size,self.screen_size-self.target_size,(2,))                
        s = self.target_size
        self.target=[x-s, y-s, x+s, y+s]
        
    def quit(self, src):
        now = time.time()
        print("Hit %d/%d targets in %.1f seconds" % (self.hits, (self.hits+self.misses), now - self.start_time))
        print("Hits per second: %.3f" % (self.hits / float(now - self.start_time)))
        print("Hit rate: %.3f" % (self.hits / float(self.hits + self.misses)))
        
    def event(self, src, event_type, event):
        if event_type=='mouseup':            
            if in_rect(self.last_filter, self.target):
                self.new_target()
                self.hits += 1
            else:
                self.misses += 1
        # exit after 10 successful clicks
        if self.hits>=10:
            src.quit(None)
        
            
                
            
        
    def draw(self, src):        
        src.clear()    
        x1,y1,x2,y2 = self.target
        
        src.rectangle(x1,y1,x2, y2, fill="grey", outline="white")
        
        # get noised position
        x,y,fx,fy,true_x,true_y,ret = self.cursor.update(src.mouse_x, src.mouse_y)
        self.last_filter = (fx,fy)
        if ret is not None:            
            src.normal(ret["mean"], ret["cov"], outline="green")
            src.text(10, 60, text="Log lik: %10.3f\n" % ret['lik'], fill="white", anchor="w")
            
        if not np.isnan(x):
            src.circle(x,y,5, fill="white")
        if not np.isnan(fx):
            src.circle(fx,fy,5, fill="green")
        now = time.time()
        
        
        src.text(10, 20, text="Hit %d/%d targets in %.1f seconds\n" % (self.hits, (self.hits+self.misses), now - self.start_time), fill="white", anchor="w")
        src.text(10, 40, text="Hits per second: %.3f\n" % (self.hits / float(now - self.start_time)), fill="white", anchor="w")
            
            
            
        
    
    