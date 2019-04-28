from tkinter import *
import numpy as np
import scipy.linalg, scipy.stats

class TKanvas(object):
    def __init__(self, draw_fn=None, tick_fn=None, event_fn=None, quit_fn=None, w=400, h=400, frame_time=20):
        self.root = Tk()
        self.canvas = Canvas(self.root, background = "black", width=w, height=h)
        self.w, self.h = w, h
        self.cx = w/2
        self.cy=h/2
        self.mouse_x = self.cx
        self.mouse_y = self.cy
        self.canvas.pack(expand=1, fill="both")        
        self.draw_fn = draw_fn
        self.tick_fn = tick_fn
        self.quit_fn = quit_fn
        self.event_fn = event_fn
        self.root.iconify()
        self.root.update()
        self.root.deiconify()
        self.root.wm_title("Canvas view: Press ESC to quit")
        self.root.bind("<Any-KeyPress>", lambda ev: self.event("keypress", ev))
        self.root.bind("<Any-KeyRelease>", lambda ev: self.event("keyrelease", ev))
        self.root.bind("<Escape>", self.quit)
        self.root.protocol("WM_DELETE_WINDOW", lambda :self.quit(None))
        self.root.bind( "<Any-Button>", lambda ev: self.event("mousedown", ev))
        self.root.bind( "<Any-ButtonRelease>", lambda ev: self.event("mouseup", ev))
        self.root.bind( "<Any-Motion>", lambda ev: self.event("mousemotion", ev))        
        self.frame_time = frame_time
        self.root.update()
        self.root.after(int(10), self.update)
        
       
    def quit(self, event):
        print("Exiting...")
        if self.quit_fn is not None:
            try:
                self.quit_fn(self)
            except:
                print("Error in quit routine; exiting anyway")
                
        self.root.destroy()
                
    def clear(self):
        self.canvas.delete(ALL)
        
    def rectangle(self, x1, y1, x2, y2, **kw):
        return self.canvas.create_rectangle( x1, y1, x2, y2, **kw)        
    
    def error_ellipse(self, mean, cov, scale=1, **kw):
        r = np.linspace(-np.pi, np.pi, 12)
        cov = scipy.linalg.sqrtm(cov)
        p = np.stack([np.cos(r), -np.sin(r)]).T
        q = np.dot(p,cov)*scale*5 + mean
        self.canvas.create_polygon(*q.ravel(), **kw)
        
    def polygon(self, pts, **kw):
        return self.canvas.create_polygon(*pts.ravel(), **kw)    
        
    def modify(self, item, **kw):
        self.canvas.itemconfig(item, **kw)
        
    def square(self, x, y, r, **kw):
        return self.rectangle(x-r, y-r, x+r, y+r, **kw)
        
    def arc(self, x1, y1, x2, y2, **kw):
        return self.canvas.create_rectangle( x1, y1, x2, y2, **kw)        
        
    def line(self, x1, y1, x2, y2, **kw ):
        return self.canvas.create_line( x1, y1, x2, y2, **kw)        
        
    def circle(self, x, y, r, **kw):
        return self.oval(x-r, y-r, x+r, y+r, **kw)
        
    def oval(self, x1, y1, x2, y2, **kw ):        
        return self.canvas.create_oval( x1, y1, x2, y2, **kw)        
        
    def text(self, x1, y1, **kw ):
        return self.canvas.create_text(x1,y1, **kw)        
        
    def move_rel(self, tagOrId, dx,dy):
        print(tagOrId, dx, dy)
        self.canvas.move( tagOrId, dx, dy)
        
    def delete(self, tagOrId):
        self.canvas.delete(tagOrId)
        
    def event(self, event_type, event):        
        if event_type=="mousemotion":
            # track mouse offset
            dx = self.mouse_x - event.x
            dy = self.mouse_y - event.y            
            event.dx = -dx
            event.dy = -dy            
            
            self.mouse_x = event.x
            self.mouse_y = event.y
            
        if self.event_fn is not None:
            self.event_fn(self, event_type, event)    
    
    def normal(self, mean, cov, ppfs=(0.65, 0.75, 0.85), **kw):
        for ppf in reversed(sorted(ppfs)):
            scale = scipy.stats.norm.ppf(ppf)
            self.error_ellipse(mean, cov, scale=scale, smooth=True, fill='', **kw)        
        
    def update(self):
        if self.draw_fn is not None:            
            self.draw_fn(self)
        self.root.update_idletasks()        
        if self.tick_fn is not None:            
            self.tick_fn(0.01)
        self.root.after(int(self.frame_time), self.update)
            
            
### Demo with mouse crosshairs
if __name__=="__main__":
    def track(src, etype, event): 
        pass        
            
    def draw(src):
        src.clear()    
        src.line(0, src.mouse_y, src.w, src.mouse_y, fill="red")        
        src.line(src.mouse_x, 0, src.mouse_x, src.h, fill="red")        
        src.normal(np.array([src.mouse_x, src.mouse_y]), np.eye(2)*30,
                   smooth=1, outline="blue", fill='')

        
        
        pass
        
    c = TKanvas(draw_fn=draw)        


    