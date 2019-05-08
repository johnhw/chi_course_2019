class Observation:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class ObservationSequence:
    def __init__(self, seq):
        self.seq = seq
        self.beam = [0.0]*len(seq) 

def ParseStimulusResponseLog(filename):
    # Initialize stimuli array and observation sequence array
    stimulus_set = []
    response_set = []
    observation_sequence_set = []

    # Read in stimulus and response sets
    fStimulusResponse = open(filename, 'r')
    stimulus_responses = fStimulusResponse.read().split('\n')
    fStimulusResponse.close()

    # N responses captured
    n_responses = len(stimulus_responses)

    valid_log_file = True
    
    if (n_responses == 1):
        print("ERROR: log file is empty")
        valid_log_file = False

    if (valid_log_file):
        # Iterate through reponses
        for response in stimulus_responses:
            # split out textual stimulus, response and set of touch points
            stimulus, response, touch_points = response.split(';')
            stimulus_set.append(stimulus)
            response_set.append(response)
            
            # Assemble ObservationSequence from touch points
            sequence = []
            touches = touch_points.split("|")
            for touch in touches:
                xPoint, yPoint = touch.split(',')
                sequence.append(Observation(float(xPoint),float(yPoint)))
            observation_sequence_set.append(ObservationSequence(sequence))

    return stimulus_set, observation_sequence_set
    
def ParseTestLog(filename):
    # Initialize observation sequence array
    observation_sequence_set = []

    # Read in stimulus and response sets
    fStimulusResponse = open(filename, 'r')
    stimulus_responses = fStimulusResponse.read().split('\n')
    fStimulusResponse.close()

    # N responses captured
    n_responses = len(stimulus_responses)

    valid_log_file = True
    
    if (n_responses == 1):
        print("ERROR: log file is empty")
        valid_log_file = False

    if (valid_log_file):
        # Iterate through reponses
        for response in stimulus_responses:
            # split out textual stimulus, response and set of touch points
            touch_points = response
                        
            # Assemble ObservationSequence from touch points
            sequence = []
            touches = touch_points.split("|")
            for touch in touches:
                xPoint, yPoint = touch.split(',')
                sequence.append(Observation(float(xPoint),float(yPoint)))
            observation_sequence_set.append(ObservationSequence(sequence))

    return observation_sequence_set