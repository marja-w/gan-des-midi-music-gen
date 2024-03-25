
import mido
import os
import logging
import re
import math
import random

import time

from datasets import generate_piano_roll

# create a class that handles processed_line tuples and generates a midi file based on the data
class MidiGenerator:
    def __init__(self, n, baseline=80, range=30, instruments=None, note_levels=None, gen2_output=None):
        self.n = n
        self.baseline = baseline
        self.range = range
        self.track = mido.MidiTrack()
        self.mid = mido.MidiFile()

        self.gen2_output = gen2_output
        self.skip_1 = int(gen2_output[0] * 20)
        if self.skip_1 == 0:
            self.skip_1 = np.random.randint(1, 10)
        self.skip_2 = int(gen2_output[1] * 20)
        if self.skip_2 == 0:
            self.skip_2 = np.random.randint(1, 10)
        self.skip_3 = int(gen2_output[2] * 20)
        if self.skip_3 == 0:
            self.skip_3 = np.random.randint(1, 10)
        self.base = int(gen2_output[3] * 100)
        if self.base == 0:
            self.base = 50
        self.tempo = int(gen2_output[4] * 10000)
        if self.tempo == 0:
            self.tempo = 50000

        self.var = int(gen2_output[5] * int(126/2))
        if self.var == 0:
            self.var = 30

        # select a key_signature based on the values in the 6th row where the values are between 0 and 1
        self.key_signature = int(gen2_output[5] * 11)
        # convert the key signature to a string
        self.key_signature = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][self.key_signature % 12]

        self.note_offsets = {}
        if note_levels is not None:
            for i,note_level in enumerate(note_levels):
                self.note_offsets[str(i)] = int(note_level)
        else:
            for i in range(0,32):
                self.note_offsets[str(i)] = random.randint(self.baseline-self.range, self.baseline+self.range)

        self.queue_lengths = {}

        self.instruments = {}
        if instruments is not None:
            for i,instrument in enumerate(instruments):
                self.instruments[str(i)] = int(instrument)
        else:
            for i in range(0,32):
                self.instruments[str(i)] = random.randint(0, 100)

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
        self.track.append(mido.MetaMessage('set_tempo', tempo=self.tempo, time=0))

        # set the time signature
        self.track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        # set the key signature
        self.track.append(mido.MetaMessage('key_signature', key=self.key_signature, time=0))

        # set the instrument
        self.track.append(mido.Message('program_change', program=0, time=0))


    def process_line(self, processed_line):
        array1, array2, array3, array4 = processed_line
        # set the instrument
        #self.track.append(mido.Message('program_change', program=array3, time=0))

        # set the note on and note off events

        #print(array1, array2, array3, array4)
        # time, event, server, arrival/departure

        # calculate midi time based on value in array1
        midi_time = max(0,int(float(array1)))


        if array4 == 'arrival' and  ( int(array2) % self.skip_1 == 0 or int(array2) % self.skip_2 == 0 or int(array2) % self.skip_3 == 0):
            if array3 in self.queue_lengths:
                self.queue_lengths[array3] += 1
            else:
                self.queue_lengths[array3] = 1


            # I NEED TO TEST HOW THIS WORKS FOR NOW AND CHANGE IT LATER IF BAD RESULTS
            queue_length = self.queue_lengths[array3]
            if 127 <= queue_length < 2*127:
                queue_length = min(127,max(0, 2*127 - queue_length))
            elif queue_length >= 2*127:
                queue_length = min(127,max(0, queue_length % 127))

            max_customer_id = self.base + self.var
            customer_id = self.base - self.var + int(array2)

            if customer_id > max_customer_id:
                customer_id = max_customer_id - ( customer_id % max_customer_id)

            self.future_events[array3] = {}
            self.future_events[array3]['time'] = midi_time
            self.future_events[array3]['velocity'] = int(customer_id) % 126
            self.future_events[array3]['service_time'] = int(queue_length)


        elif array4 == 'departure' and  ( int(array2) % self.skip_1 == 0 or int(array2) % self.skip_2 == 0 or int(array2) % self.skip_3 == 0):

            if array3 in self.future_events:
                # change the instrument to the instrument of the server
                on_time = max(0, int(self.future_events[array3]['time']))
                #self.track.append(mido.Message('program_change', program=self.instruments[array3], time=on_time))
                self.track.append(mido.Message('note_on', channel=0, note=self.note_offsets[array3], velocity=int(self.future_events[array3]['velocity']),  time=on_time))

                # change the instrument to the instrument of the server
                off_time = max(0,int(self.future_events[array3]['time'] + (midi_time-self.future_events[array3]['time']) + max(0,self.future_events[array3]['service_time']))) 
                #self.track.append(mido.Message('program_change', program=self.instruments[array3], time=off_time))
                self.track.append(mido.Message('note_off', channel=0, note=self.note_offsets[array3], velocity=self.future_events[array3]['velocity'],  time=off_time))


            if array3 in self.queue_lengths:
                self.queue_lengths[array3] -= 1
            else:
                self.queue_lengths[array3] = 0

        elif array4 == 'processing' and  ( int(array2) % self.skip_1 == 0 or int(array2) % self.skip_2 == 0 or int(array2) % self.skip_3 == 0):
            self.future_events[array3]['service_time'] += midi_time


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

def process_adjsim_log(n=5000, baseline=70, range=50, instruments=np.arange(0,16), note_levels=np.random.randint(0, 127, 16), gen2_output=None, count=0):
    # Example usage:
    log_processor = LogLineProcessor(r"INFO:root:([0-9]*\.[0-9]+|[0-9]+) - ([0-9]*\.[0-9]+|[0-9]+) - ([0-9]*\.[0-9]+|[0-9]+) - (arrival|departure)")

    count = 0
    max = 5000

    midi_generator = MidiGenerator(n=max, baseline=baseline, range=range, instruments=instruments, note_levels=note_levels, gen2_output=gen2_output)

    if gen2_output is None:
        gen2_output = np.random.rand(20)

    # Read the log file line by line
    with open("./logs/simulation.log", 'r') as f:
        for line in f:
            count += 1
            if count > max:
                break
            processed_line = log_processor.process_line(line)
            if processed_line:
                midi_generator.process_line(processed_line)

    if count % 10 == 0:
        # save the midi file
        mido.save('adj_sim_outputs/midi/simulation_{}.mid'.format(count), midi_generator.mid)

    return generate_piano_roll(midi_generator.mid)