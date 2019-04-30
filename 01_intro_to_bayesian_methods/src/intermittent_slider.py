import tkanvas
from tkanvas import TKanvas
import numpy as np
import time

class Box:
    def __init__(self, name, left, right, prior, color):
        self.name = name
        self.left = left
        self.right = right
        self.prior = prior
        self.color = color

    
class SliderDemo(object):
    def __init__(self, boxes):
        self.boxes = boxes        
        self.screen_size = 800
        self.slider_height = 50        
        self.canvas = TKanvas(draw_fn=self.draw, event_fn=self.event,
         quit_fn=self.quit, w=self.screen_size, h=self.slider_height)     
        self.last_x = None
        self.last_t = time.time()
        self.dx = 0
        self.x = 0

    def quit(self, src):
        return

    def event(self, src, event_type, event):
   
        pass
                                                
    def draw(self, src):        
        src.clear()    
        screen_x = src.mouse_x
        

        t = time.time()
        self.x = screen_x / self.screen_size
        
        if self.last_x!=None:
            dx = self.x - self.last_x
            dt = t - self.last_t
            self.dx = dx / dt
        
        self.last_x = self.x
        self.last_t = t

        for box in self.boxes:
            screen_left = box.left * self.screen_size
            screen_right = box.right * self.screen_size
            src.rectangle(screen_left, 0, screen_right, self.slider_height, fill=box.color)

        src.line(screen_x, 0, screen_x, self.slider_height, fill="grey")

        
        observed = np.random.uniform(0,1)<0.05

        if observed:
            src.line(screen_x, 0, screen_x, self.slider_height, fill="white", width=5)
            


if __name__=="__main__":
    boxes = [Box("A", 0.0, 0.1, prior=0.25, color='red'), 
         Box("X", 0.1, 0.75, prior=0.25, color='black'), 
         Box("B", 0.9, 1.0, prior=0.5, color='yellow')]
    slider = SliderDemo(boxes)
    slider.canvas.root.mainloop()

            
        
    
    