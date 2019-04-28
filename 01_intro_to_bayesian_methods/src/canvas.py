# Canvas module for simple drawing and mouse handling
# Quintin Cutts
# Last modified 24 - 11 - 08

###  IF YOU JUST WANT TO KNOW WHAT FUNCTIONS ARE AVAILABLE
###  THEN JUMP STRAIGHT TO THE END OF THIS CODE - YOU'LL
###  FIND THE FULL LIST THERE.

### --------------------------------------------------------

from Tkinter import *
import threading
import time
import exceptions
import sys

class WindowGone(exceptions.Exception):
    def __init__(self, args=[]):
        self.args = args

# These are all the inner Glasgow Canvas functions
class RawCanvas:
    def __init__(self):
        self.mainThread = threading.currentThread()
        self._events = []
        self.mainLoopRunning = False
        self.no_current_keyhandler_call = True    # Concurrency control - stops multiple simultaneous calls of the handler
        self.no_current_mousehandler_call = True

    # These are the main drawing functions - calling straight through to the
    # underlying Tkinter Canvas functions
    def create_rectangle( self, x1, y1, x2, y2, *kw ):
        r = self._canvas.create_rectangle( x1, y1, x2, y2, kw )
        self._canvas._root().update()
        return r
    def create_arc( self, x1, y1, x2, y2, *kw ):
        r = self._canvas.create_arc( x1, y1, x2, y2, kw )
        self._canvas._root().update()
        return r        
    def create_line( self, x1, y1, x2, y2, *kw ):
        r = self._canvas.create_line( x1, y1, x2, y2, kw)
        self._canvas._root().update() 
        return r       
    def create_oval( self, x1, y1, x2, y2, *kw ):
        r = self._canvas.create_oval( x1, y1, x2, y2, kw )
        self._canvas._root().update()
        return r        
    def create_text( self, x1, y1, *kw ):
        r = self._canvas.create_text( x1, y1, kw )
        self._canvas._root().update() 
        return r       
    def move( self, tagOrId, xInc, yInc ):
        self._canvas.move( tagOrId, xInc, yInc )
        self._canvas._root().update()        
    def delete( self, tagOrId ):
        self._canvas.delete( tagOrId )
        self._canvas._root().update()
    def set_title( self, t ):
        self._canvas._root().title( t )
        self._canvas._root().update()

    # These two both set the mainloop running
    # For this one, it's just to set a button handler to kill the window when pressed
    def _normal_complete( self, m = "Click mouse to end" ):
        global _can
        self._canvas.unbind("<Button-1>")
        self._canvas.bind("<Button-1>", _can.destroy)
        wait( 0.5 )
        self._canvas._root().title( m )
        self._canvas._root().update()
        self.run()
    # and for this one, it sets the mainloop running alone, trusting the user has
    # set some callbacks already.
    def run( self ):
        if not self.mainLoopRunning:
            self.mainLoopRunning = True
            try:
                #self._canvas._root().mainloop()
                pass
            except WindowGone:
                pass   

    # These three functions all set flags in the _events list - and then are handled
    # by the originating tkinter thread later.  Required as if the code to execute
    # these functions called by the non-Tkinter thread, then Tkinter hangs.
    def set_size( self, x, y ):
        self._events = self._events + [ ["ss",x,y] ]
    def complete( self, a=None ):
        if threading.currentThread() != self.mainThread:
            if "c" not in self._events:
                if a == None:
                    self._events = self._events + ["c"]
                else:
                    self._events = self._events + ["c"+a]
        else:
            if a == None:
                self._normal_complete()
            else:
                self._normal_complete( a )
    def quitCanvas( self ):
        if "q" not in self._events:
            self._events = self._events + [ "q" ]

    # Enables a separate thread to be run alongside the Tkinter thread.
    # This is the unsafest part of the module, since separate threads shouldn't be allowed
    # to call the Tkinter functions - but it seems to work for the Canvas functions
    def runGraphicsFn( self,g ):
        def gWrap():
            try:
                g()
            except WindowGone:   # Enables threads to die quietly if Tkinter closed by user
                pass
        newThread = threading.Thread( target = gWrap )
        newThread.start()

    # A range of event handler setting functions next
    def set_keydown_handler( self, handler ):
        def inner_handler( e ):
            if self.no_current_keyhandler_call:
                self.no_current_keyhandler_call = False
                handler( e.keysym )
                self.no_current_keyhandler_call = True
        self._canvas._root().bind( "<Any-KeyPress>", inner_handler )
        self._canvas._root().update()
    def unset_keydown_handler( self ):
        self._canvas._root().unbind( "<Any-KeyPress>" )
    def set_mousedown_handler( self, handler ):
        def inner_handler( e ):
            if self.no_current_mousehandler_call:
                self.no_current_mousehandler_call = False
                handler( e.x, e.y, e.num )
                self.no_current_mousehandler_call = True
        self._canvas.bind( "<Any-Button>", inner_handler )
        self._canvas._root().update()
    def unset_mousedown_handler( self ):
        self._canvas._root().unbind( "<Any-Button>" )
    def set_mouseup_handler( self, handler ):
        def inner_handler( e ):
            if self.no_current_mousehandler_call:
                self.no_current_mousehandler_call = False
                handler( e.x, e.y, e.num )
                self.no_current_mousehandler_call = True
        self._canvas.bind( "<Any-ButtonRelease>", inner_handler )
        self._canvas._root().update()
    def unset_mouseup_handler( self ):
        self._canvas._root().unbind( "<Any-ButtonRelease>" )
    def set_mousemotion_handler( self, handler ):
        def inner_handler( e ):
            if self.no_current_mousehandler_call:
                self.no_current_mousehandler_call = False
                handler( e.x, e.y )
                self.no_current_mousehandler_call = True
        self._canvas.bind( "<Motion>", inner_handler )
        self._canvas._root().update()
    def unset_mousemotion_handler( self ):
        self._canvas._root().unbind( "<Motion>" )

_can = None            # This is the Glasgow canvas
_hadCan = False        # Did we ever open a Canvas, even though it might now be dead?
_blockCalls = False    # When True, don't try to execute Canvas ops, because Window has been closed

#import IPython.kernel
class Can( RawCanvas ):
    def __init__( self, draw_fn=None, tick_fn= None):
        global _root, _canvas
        self._root = Tk()
        self._canvas = Canvas( self._root, background = "white" )
        self._canvas.pack(expand=1, fill="both" )
        RawCanvas.__init__( self )
        self.draw_fn = draw_fn
        self.tick_fn = tick_fn
        self._root.iconify()
        self._root.update()
        self._root.deiconify()
        self._root.update()

        def destroy( event=None, extra=None ):
            pass
            #global _blockCalls, _root
            #_blockCalls = True            
            #time.sleep(0.5)
            #self._root.destroy()

        self.destroy = destroy
        self._root.protocol("WM_DELETE_WINDOW", self.destroy)

        # Finally, get the event checker running, to pick up events
        # coming in from other threads that want to act on the tkinter thread
        def update_tkinter():
            if _blockCalls:
                return
            
            if self.draw_fn:
                self.draw_fn(self._canvas)
            if self.tick_fn:
                result = self.tick_fn(0.01)            
                if result:                                        
                    self._root.destroy()
                    
                    
            if self._events != []:
                for e in self._events:
                    if type( e ) == type( "" ):
                        if e[0] == "c":
                            if len( e ) == 1:
                                self._normal_complete()
                            else:
                                self._normal_complete( e[1:] )
                        elif e == "q":
                            self.destroy()
                    else:   # must be ["ss", x, y] for a set screen
                        self._canvas.config( width = e[1], height = e[2] )
         
                self._events = []
            self._root.after( 10, update_tkinter )
        update_tkinter()

def _getCanvas():
    global _can, _hadCan, _blockCalls
    if (_hadCan and not _can) or _blockCalls:
        raise WindowGone
    if not _can:
        _can = Can()
        _hadCan = True
    return _can

##########################################################
# These are the only visible functions out of the module

def create_rectangle( x1, y1, x2, y2, **kw ):
    return _getCanvas().create_rectangle( x1, y1, x2, y2, kw )
def create_arc( x1, y1, x2, y2, **kw ):
    return _getCanvas().create_arc( x1, y1, x2, y2, kw )
def create_line( x1, y1, x2, y2, **kw ):
    return _getCanvas().create_line( x1, y1, x2, y2, kw )
def create_oval( x1, y1, x2, y2, **kw ):
    return _getCanvas().create_oval( x1, y1, x2, y2, kw )
def create_text( x1, y1, **kw ):
    return _getCanvas().create_text( x1, y1, kw )
def move( tagOrId, xInc, yInc ):
    _getCanvas().move( tagOrId, xInc, yInc )
def wait( t1 ):
    time.sleep( t1 )
def delete( tagOrId ):
    _getCanvas().delete( tagOrId )
def set_title( txt ):
    _getCanvas().set_title( txt )
def set_size( x, y ):
    _getCanvas().set_size( x, y )
def complete( a = None ):
    _getCanvas().complete( a )
def run():
    _getCanvas().run()
def quitCanvas():
    _getCanvas().quitCanvas()
def runGraphicsFn( g ):
    _getCanvas().runGraphicsFn( g )
def set_keydown_handler( handler ):
    _getCanvas().set_keydown_handler( handler )
def unset_keydown_handler():
    _getCanvas().unset_keydown_handler()
def set_mousedown_handler( handler ):
    _getCanvas().set_mousedown_handler( handler )
def unset_mousedown_handler( handler ):
    _getCanvas().unset_mousedown_handler()
def set_mouseup_handler( handler ):
    _getCanvas().set_mouseup_handler( handler )
def unset_mouseup_handler():
    _getCanvas().unset_mouseup_handler()
def set_mousemotion_handler( handler ):
    _getCanvas().set_mousemotion_handler( handler )
def unset_mousemotion_handler():
    _getCanvas().unset_mousemotion_handler()
