import datetime
import math
import numpy as np
from tkinter import *

# Key class, as used in from tutorial
class Key:
    def __init__(self, centre_x, centre_y, letter):
        self.centre_x = centre_x
        self.centre_y = centre_y
        self.letter = letter
         
# Tkinter application class         
class ToyKeyboard():
    
    # Animate keyboard key press at x,y
    def animate_key(self,x,y):
        # Delete existing key highlight
        self.layoutCanvas.delete(self.keyHighlight)
    
        # Determine closest key to touch points
        iClosestKey = 0
        numKeys = len(self.keys)
        testKey = self.keys[iClosestKey]
        minKeyOffset = math.hypot(x - testKey.centre_x,y - testKey.centre_y)
        for iKey in range(1,numKeys):
            testKey = self.keys[iKey]
            keyOffset = math.hypot(x - testKey.centre_x,y - testKey.centre_y)
            if (keyOffset < minKeyOffset):
                minKeyOffset = keyOffset
                iClosestKey = iKey
        
        # Closest key has been found
        closestKey = self.keys[iClosestKey]

        # Print out closest key letter
        updatedReponse = self.currentResponse.get() + closestKey.letter.upper()
        self.currentResponse.set(updatedReponse)
        
        # Draw a simple circle at the key closest to the touch point
        keyRadius = 50
        highlightWidth = 2
        x0 = closestKey.centre_x - keyRadius
        y0 = closestKey.centre_y - keyRadius
        x1 = closestKey.centre_x + keyRadius
        y1 = closestKey.centre_y + keyRadius
        self.keyHighlight = self.layoutCanvas.create_oval(x0,y0,x1,y1, width=highlightWidth, outline="#6699ff")

    # Animate touch point at x,y
    def animate_touch(self,x,y):
        self.layoutCanvas.delete(self.touchPoint)

        # Draw a simple circle to indicate where the touch occured
        touchPointRadius = 5
        x0 = x - touchPointRadius
        y0 = y - touchPointRadius
        x1 = x + touchPointRadius
        y1 = y + touchPointRadius
        self.touchPoint = self.layoutCanvas.create_oval(x0,y0,x1,y1, fill="red")
    
    # Callback on mouse button click on the main canvas
    def on_click(self,event):
        # Animate the closest key and touch point
        self.animate_key(event.x,event.y)
        self.animate_touch(event.x,event.y)
        
        # Normalize touch point and log to active log file
        xTouch = (event.x - 150.0) / 100.0
        yTouch = (event.y - 150.0) / 100.0
        
        # Append normalized touch point to history
        self.touch_points = np.append(self.touch_points, np.array([[xTouch,yTouch]]), axis=0)
        
    # Callback on Next button press
    def on_next(self):
        # Log current stimulus and response to file
        self.log_stimulus_reponse()
        # Reset the interface appearance
        self.reset()
    
        # Retrieve next stimulus and update progress, or quit
        self.i_word += 1
        if self.i_word < self.n_stimuli:
            self.progressString.set("%d/%d" % (self.i_word + 1, self.n_stimuli))
            self.currentStimulus.set(self.stimulus_words[self.i_word].upper())
        else:
            self.quit()
            
    def log_stimulus_reponse(self):
        logString = ""
        pointLog = ""
        nPoints = self.touch_points.shape[0]
        for i in range(0,nPoints):
            pointLog = pointLog + "{1:.3f},{2:.3f}".format(pointLog,self.touch_points[i,0],self.touch_points[i,1])
            if i < nPoints - 1:
                pointLog = pointLog + "|"
        if (self.i_word != 0):
            logString = "\n"
        logString = logString + "%s;%s;%s" % (self.currentStimulus.get().lower(),self.currentResponse.get().lower(),pointLog)

        # Reset touch point history
        self.touch_points = np.empty((0,2), int)
        self.logFile.write(logString)
        
    # Callback on Reset button press
    def reset(self):
        self.layoutCanvas.delete(self.keyHighlight)
        self.layoutCanvas.delete(self.touchPoint)
        self.currentResponse.set("")
        
    def quit(self):
        self.root.destroy()
    
    # Create a new log file with a unique filename, if required uncomment timeStamp
    def create_new_log(self):
        #timeStamp = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
        #filename = "stimulus_response_{0}.csv".format(timeStamp)
        filename = "stimulus_response.log"
        self.logFile = open(filename,"w")
        print("logging touches to: {0}".format(filename))

    # Close the active log file      
    def close_log(self):
        self.logFile.close()
        
    # Initialize the Tkinter application        
    def __init__(self, master=None):

        self.root = Tk()
        
        #Frame.__init__(self, self.root)
        
        self.root.title("ABCD Keyboard")
        
        # Read in stimulus phrases
        fStimulus = open('stimulus.txt', 'r')
        self.stimulus_words = fStimulus.read().split('\n')
        
        # Shuffle words
        np.random.shuffle(self.stimulus_words)
        
        # Restrict number of stimuli shown to 10
        self.n_stimuli = 10 #len(self.stimulus_words)
        self.i_word = 0
        
        # Initialize array to track touch point history
        self.touch_points = np.empty((0,2), int)
                
        # Fix the keyboard size, add significant padding to allow for collection of erroneous touches
        self.root.minsize(width=400, height=400)
        self.root.maxsize(width=400, height=400)
        #master.minsize(width=400, height=400)
        #master.maxsize(width=400, height=400)
                 
        # Add a canvas, with an image that is the keys, significantly simplifies the capture of touch locations
        self.layoutCanvas = Canvas(master,width=400, height=400)
        self.layoutCanvas.create_rectangle(0, 0, 400, 400, fill="white")
        self.LayoutImage = PhotoImage(file = './key_layout.gif')
        self.layoutCanvas.create_image(200, 200, image=self.LayoutImage)
        self.layoutCanvas.bind("<ButtonPress-1>",self.on_click)

        # Stimulus label
        self.stimulusLabel = Label(self.layoutCanvas, text = "STIMULUS:")
        self.stimulusLabel.configure(width = 10, bg = "white", relief = FLAT, anchor = W)
        self.stimulusLabel_window = self.layoutCanvas.create_window(100, 25, anchor=NW, window=self.stimulusLabel)

        # Current stimulus text        
        self.currentStimulus = StringVar()
        self.currentStimulus.set(self.stimulus_words[self.i_word].upper())
        self.stimulusText = Label(self.layoutCanvas, textvariable = self.currentStimulus)
        self.stimulusText.configure(width = 17, bg = "white", relief = GROOVE, anchor = W)
        self.stimulusText_window = self.layoutCanvas.create_window(300, 25, anchor=NE, window=self.stimulusText)
                   
        # Response label
        self.responseLabel = Label(self.layoutCanvas, text = "RESPONSE:")
        self.responseLabel.configure(width = 10, bg = "white", relief = FLAT, anchor = W)
        self.responseLabel_window = self.layoutCanvas.create_window(100, 50, anchor=NW, window=self.responseLabel)

        # Current response text
        self.currentResponse = StringVar()
        self.currentResponse.set("")
        self.responseText = Label(self.layoutCanvas, textvariable = self.currentResponse)
        self.responseText.configure(width = 17, bg = "white", relief = GROOVE, anchor = W)
        self.responseText_window = self.layoutCanvas.create_window(300, 50, anchor=NE, window=self.responseText)

        # Progress feedback
        self.progressString = StringVar()
        self.progressString.set("1/%d" % (self.n_stimuli))
        self.progressText = Label(self.layoutCanvas, textvariable = self.progressString)
        self.progressText.configure(width = 10, bg = "white", relief = FLAT)
        self.progressText_window = self.layoutCanvas.create_window(5, 345, anchor=NW, window=self.progressText)
        
        # Add a next button to the keyboard for progress to next stimulus        
        self.nextButton = Button(self.layoutCanvas, text = "NEXT", command = self.on_next)
        self.nextButton.configure(width = 10, activebackground = "#33B5E5", relief = FLAT)
        self.nextButton_window = self.layoutCanvas.create_window(5, 395, anchor=SW, window=self.nextButton)
                
        # Add a quit button that will exit the application immediately        
        self.quitButton = Button(self.layoutCanvas, text = "QUIT", command = self.quit)
        self.quitButton.configure(width = 10, activebackground = "#33B5E5", relief = FLAT)
        self.quitButton_window = self.layoutCanvas.create_window(395, 395, anchor=SE, window=self.quitButton)
        
        # Pack the canvas        
        self.layoutCanvas.pack()
        
        # Create a new log file to caputre observation sequence
        self.create_new_log()
               
        # Initialize drawing objects for use in animation of key press and touch point
        self.touchPoint = self.layoutCanvas.create_oval(0,0,0,0)
        self.keyHighlight = self.layoutCanvas.create_oval(0,0,0,0)

        # Specify key layout for use in finding close key to touch location
        a_key = Key(150,150,'a')
        b_key = Key(150,250,'b')
        c_key = Key(250,150,'c')
        d_key = Key(250,250,'d')
        self.keys = [a_key, b_key, c_key, d_key]
        
        # Use this trick to bring keyboard to front
        self.root.iconify()
        self.root.update()
        self.root.deiconify()
