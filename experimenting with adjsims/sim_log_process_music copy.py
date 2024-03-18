
import mido
import os
import logging
import re
import math
import random

# create a class that handles processed_line tuples and generates a midi file based on the data
class MidiGenerator:
    def __init__(self, n, baseline=80, range=30, instruments=[], note_levels=[]):
        self.n = n
        self.baseline = baseline
        self.range = range
        self.track = mido.MidiTrack()
        self.mid = mido.MidiFile()

        self.note_offsets = {}
        for i,note_level in enumerate(note_levels):
            self.note_offsets[i] = note_level

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
        midi_time = max(0,int(float(array1)))

        velocity = (int(array2) + 30) % 126

        note = 0

        if 127 < int(array2) < 2*127:
            note = 2*127 - int(array2)
        else:
            note = int(array2) % 127


        # change midi channel based on server for different instruments
        if array3 not in self.instruments:
            self.instruments[array3] = random.randint(0, 100)
        self.track.append(mido.Message('program_change', program=self.instruments[array3], time=midi_time))

        # change note offset based on server
        if array3 not in self.note_offsets:
            #self.note_offsets[array3] = random.randint(self.baseline+30, self.baseline+30)
            self.note_offsets[array3] = random.randint(40, 126)

        process_method = 2

        if array4 == 'arrival' and  ( int(array2) % 3 == 0  ):
            if array3 in self.queue_lengths:
                self.queue_lengths[array3] += 1
            else:
                self.queue_lengths[array3] = 1

            if process_method == 2 or process_method == 0:
                velocity = self.queue_lengths[array3] + self.note_offsets[array3]
                if 127 < velocity < 2*127:
                    velocity = abs((2*127 - velocity) % 127)
                else:
                    velocity = abs(velocity % 127)

                self.future_events[array3] = {}
                self.future_events[array3]['time'] = midi_time
                self.future_events[array3]['velocity'] = velocity
                self.future_events[array3]['service_time'] = 0

            if process_method == 1:
                self.track.append(mido.Message('note_on', channel=0, note=int(array2)%127, velocity=velocity,  time=midi_time))
            elif process_method == 3:
                self.track.append(mido.Message('note_on', channel=0, note=self.note_offsets[array3], velocity=note,  time=midi_time))

        elif array4 == 'departure' and  ( int(array2) % 3 == 0  ):

            if array3 in self.future_events:
                if process_method == 2 or process_method == 0:
                    self.track.append(mido.Message('note_on', channel=0, note=self.note_offsets[array3], velocity=self.future_events[array3]['velocity'], time=self.future_events[array3]['time'])) 
                    #self.track.append(mido.Message('note_off', channel=0, note=self.note_offsets[array3], velocity=velocity, time=midi_time-self.future_events[array3]['time']))
            else:
                if process_method == 2 or process_method == 0: # seperate if statement for later adding other options... maybe
                    self.track.append(mido.Message('note_on', channel=0, note=self.note_offsets[array3], velocity=velocity, time=midi_time))

            if array3 in self.queue_lengths:
                self.queue_lengths[array3] -= 1
            else:
                self.queue_lengths[array3] = 0

            if array3 in self.future_events and process_method == 2:
                self.track.append(mido.Message('note_off', channel=0, note=self.note_offsets[array3], velocity=self.future_events[array3]['velocity'], time=self.future_events[array3]['time']+self.future_events[array3]['service_time']*300))
            else:
                if array3 in self.future_events and process_method == 0:
                    self.track.append(mido.Message('note_off', channel=0, note=self.note_offsets[array3], velocity=self.future_events[array3]['velocity'], time=midi_time))
                elif process_method == 1:
                    self.track.append(mido.Message('note_off', channel=0, note=int(array2)%127, velocity=velocity, time=midi_time))
                elif process_method == 3:
                    self.track.append(mido.Message('note_off', channel=0, note=self.note_offsets[array3], velocity=note, time=midi_time))
                else:
                    self.track.append(mido.Message('note_off', channel=0, note=self.note_offsets[array3], velocity=velocity, time=midi_time))
                    print('Error: future event not found')

        elif array4 == 'processing' and  ( int(array2) % 3 == 0 ) and process_method == 2:
            self.future_events[array3]['service_time'] = midi_time

    def save_midi(self, filename='output.mid'):
        # add the end of track message
        self.track.append(mido.MetaMessage('end_of_track'))

        # add the track to the midi file
        self.mid.tracks.append(self.track)

        # save the midi file
        self.mid.save(filename)



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

def process_adjsim_log(n=5000, baseline=70, range=50, instruments=np.arange(0,16), note_levels=np.random.randint(0, 127, 16)):
    # Example usage:
    log_processor = LogLineProcessor(r"INFO:root:([0-9]*\.[0-9]+|[0-9]+) - ([0-9]*\.[0-9]+|[0-9]+) - ([0-9]*\.[0-9]+|[0-9]+) - (arrival|departure)")

    count = 0
    max = 5000

    midi_generator = MidiGenerator(n=max, baseline=baseline, range=range, instruments=instruments, note_levels=note_levels)


    # Read the log file line by line
    with open('logs/simulation.log', 'r') as f:
        for line in f:
            count += 1
            if count > max:
                break
            processed_line = log_processor.process_line(line)
            if processed_line:
                midi_generator.process_line(processed_line)

    filepath = 'adj_sim_outputs\midi\output.mid' # need to change this if we want to keep the original midi files, not necessary for now

    # save the output midi to /adj_sim_output/midi/output.mid
    midi_generator.save_midi(filename=filepath) 

    return filepath 