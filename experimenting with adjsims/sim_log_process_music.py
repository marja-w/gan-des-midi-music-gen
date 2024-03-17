
import mido
import os
import logging
import re
import math
import random

# create a class that handles processed_line tuples and generates a midi file based on the data
class MidiGenerator:
    def __init__(self, n, baseline=80, range=30, instruments=[]):
        self.n = n
        self.baseline = baseline
        self.range = range
        self.track = mido.MidiTrack()
        self.mid = mido.MidiFile()

        self.note_offsets = {}

        self.queue_lengths = {}

        self.instruments = {}
        for i,instrument in enumerate(instruments):
            self.instruments[i] = instrument

        self.future_events = {}

    def generate_midi(self):
        
        # create a midi file based on the data

        #for now, create a basic mido midi file where 
        # array4 is note on and note off events for arrival and departure
        # array3 is the instrument for the note on and note off events
        # array2 is the velocity for the note on and note off events
        # array 1 is the current time

        # create a new midi file
        
        self.mid.tracks.append(self.track)

        # set the tempo
        self.track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

        # set the time signature
        self.track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        # set the key signature
        self.track.append(mido.MetaMessage('key_signature', key='C', time=0))


    def process_line(self, processed_line):
        array1, array2, array3, array4 = processed_line
        # set the instrument
        #self.track.append(mido.Message('program_change', program=array3, time=0))

        # set the note on and note off events

        """
        TO-DO/IDEAS FOR FUTURE CHANGES:
        - save data about the queue lengths in a dictionary for when an event arrives at a server
        - when an event is being processed by the server, then add note_on to the midi file
        - need some events to be quiet and not result in a note_on or note_off event to be added to the midi file
        - when an event is finished being processed by the server, then add note_off to the midi file
        """

        #print(array1, array2, array3, array4)
        # time, event, server, arrival/departure


        # calculate midi time based on value in array1
        midi_time = int(float(array1))

        velocity = int(array2) % 126


        # change midi channel based on server for different instruments
        if array3 not in self.instruments:
            self.instruments[array3] = random.randint(0, 100)
        self.track.append(mido.Message('program_change', program=self.instruments[array3], time=midi_time))

        # change note offset based on server
        if array3 not in self.note_offsets:
            #self.note_offsets[array3] = random.randint(self.baseline+30, self.baseline+30)
            self.note_offsets[array3] = random.randint(40, 126)
   
        if array4 == 'arrival' and  ( int(array2) % 3 == 0 or int(array2) % 5 == 0 ):
            if array3 in self.queue_lengths:
                self.queue_lengths[array3] += 1
            else:
                self.queue_lengths[array3] = 1

            self.future_events[array3] = {}
            self.future_events[array3]['time'] = midi_time
            self.future_events[array3]['velocity'] = velocity
            self.future_events[array3]['service_time'] = 0

            #self.track.append(mido.Message('note_on', channel=0, note=self.note_offsets[array3], velocity=velocity,  time=midi_time)) # time needs to be changed to something else... maybe server processing time?
            self.track.append(mido.Message('note_on', channel=0, note=int(array2)%127, velocity=velocity,  time=midi_time))

        elif array4 == 'departure' and  ( int(array2) % 3 == 0 or int(array2) % 5 == 0 ):

            #if array3 in self.future_events:
                #self.track.append(mido.Message('note_on', channel=0, note=self.note_offsets[array3], velocity=self.future_events[array3]['velocity'], time=midi_time-self.future_events[array3]['time']))  # time needs to be changed to something else... maybe server processing time?

            if array3 in self.queue_lengths:
                self.queue_lengths[array3] -= 1
            else:
                self.queue_lengths[array3] = 0
            self.track.append(mido.Message('note_off', channel=0, note=int(array2)%127, velocity=velocity, time=midi_time))  # time needs to be changed to something else... maybe server processing time?
            #self.track.append(mido.Message('note_off', channel=0, note=self.note_offsets[array3], velocity=self.future_events[array3]['velocity'], time=self.future_events[array3]['time']+self.future_events[array3]['service_time']*2))

        elif array4 == 'processing' and  ( int(array2) % 3 == 0 or int(array2) % 5 == 0 ):
            self.future_events[array3]['service_time'] = midi_time

    def save_midi(self):
        # add the end of track message
        self.track.append(mido.MetaMessage('end_of_track'))

        # add the track to the midi file
        self.mid.tracks.append(self.track)

        # save the midi file
        self.mid.save('output.mid')



class LogLineProcessor:
    def __init__(self, regex_format):
        self.regex_format = regex_format

    def process_line(self, line):
        match = re.match(self.regex_format, line)
        if match:
            return match.group(1), match.group(2), match.group(3), match.group(4)
        else:
            return None

import numpy as np

def process_adjsim_log(n=2000, baseline=70, range=50, instruments=np.arange(0,16)):
    # Example usage:
    log_processor = LogLineProcessor(r"INFO:root:([0-9]*\.[0-9]+|[0-9]+) - ([0-9]*\.[0-9]+|[0-9]+) - ([0-9]*\.[0-9]+|[0-9]+) - (arrival|departure)")

    count = 0
    max = 2000

    midi_generator = MidiGenerator(n=max, baseline=baseline, range=range, instruments=instruments)


    # Read the log file line by line
    with open('logs/simulation.log', 'r') as f:
        for line in f:
            count += 1
            if count > max:
                break
            processed_line = log_processor.process_line(line)
            if processed_line:
                midi_generator.process_line(processed_line)


    midi_generator.save_midi()