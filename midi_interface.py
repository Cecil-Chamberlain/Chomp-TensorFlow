from mido.ports import MultiPort
import mido
import time

output = [0 for x in range(88)]

class MIDIDevice(object):
    def __init__(self, training=0):
        try:
            devices = mido.get_output_names()
            for i, d in enumerate(devices):
                print("{}: {}".format(i, d))
            i = input("Enter number of chosen device:")
            self.device = devices[int(i)]
            print("Device opened =", self.device)
            self.inport = mido.open_input(self.device)
            self.outport = mido.open_output(self.device)
        except IndexError:
            print("Error opening MIDI Device.")
        self.threshold = 0.1
        self.held_notes = []
        self.training = training
        self.current_note = '-'

    def send(self, _input):
        if self.training:
            # Training
            for frame in _input:
                for i, j in enumerate(frame): # for i, j in enumerate(step):
                    if j > self.threshold and i not in self.held_notes:
                        self.held_notes.add(i)
                        self.outport.send(mido.Message('note_on', note=i+21, velocity= int(j * 127), time=0))
                    if j <= self.threshold and i in self.held_notes:
                        self.held_notes.remove(i)
                        self.outport.send(mido.Message('note_off', note=i+21, velocity=0))
                #time.sleep(1/12)

    def receive(self):
        x = [i for i in self.inport.iter_pending()]
        if len(x) > 0:
            for i in x:
                if i.velocity > 0:
                    self.held_notes.append(i.note)
                else:
                    self.held_notes.remove(i.note)
            if len(self.held_notes) > 0:
                self.current_note = self.held_notes[-1]
            else:
                self.current_note = '-'
        return str(self.current_note)
        
