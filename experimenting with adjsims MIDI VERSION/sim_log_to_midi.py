
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
        self.skip_1 = max(2,int(gen2_output[0] * 10))
        if self.skip_1 == 0:
            self.skip_1 = np.random.randint(1, 10)
        self.skip_2 = max(2, int(gen2_output[1] * 10))
        if self.skip_2 == 0:
            self.skip_2 = np.random.randint(1, 10)
        self.skip_3 = max(2, int(gen2_output[2] * 10))
        if self.skip_3 == 0:
            self.skip_3 = np.random.randint(1, 10)
        self.base = int(gen2_output[3] * 90)
        if self.base < 50:
            self.base = 80
        self.tempo = min(int(gen2_output[4] * 1000000), 16777215)
        if self.tempo == 0:
            self.tempo = 500000

        self.var = int(gen2_output[5] * int(126/2))
        if self.var == 0:
            self.var = 30

        # select a key_signature based on the values in the 6th row where the values are between 0 and 1
        self.key_signature = int(gen2_output[5] * 11)
        # convert the key signature to a string
        self.key_signature = ['C', 'C#', 'D', 'E', 'F', 'F#', 'G', 'G#m', 'A', 'A#m', 'B'][self.key_signature % 11]

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

        self.generate_midi()

        self.previous_time = 0

    def generate_midi(self):
        
        # create a midi file based on the data

        #for now, create a basic mido midi file where 
        # array4 is note on and note off events for arrival and departure
        # array3 is the instrument for the note on and note off events
        # array2 is the velocity for the note on and note off events
        # array 1 is the current time

        # create a new midi file

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


        if midi_time < 200 and len(self.track) < 500:
            
            #TO-DO THIS IS A BIT OF A WORK AROUND.... SHOULD NOT BE NEEDED ( Think simulator is generating negative times for some distributions)
            if self.previous_time > midi_time:
                midi_time = self.previous_time
            

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
                self.future_events[array3]['time'] = int(midi_time)
                self.future_events[array3]['velocity'] = int(customer_id) % 126
                self.future_events[array3]['service_time'] = int(queue_length)

                # change the instrument to the instrument of the server
                on_time = int(max(self.previous_time, int(self.future_events[array3]['time'])))
                self.previous_time = on_time
                #self.track.append(mido.Message('program_change', program=self.instruments[array3], time=on_time))
                self.track.append(mido.Message('note_on', channel=0, note=int(self.note_offsets[array3]), velocity=int(self.future_events[array3]['velocity']),  time=on_time))


            elif array4 == 'departure' and  ( int(array2) % self.skip_1 == 0 or int(array2) % self.skip_2 == 0 or int(array2) % self.skip_3 == 0):

                if array3 in self.future_events:
                    # change the instrument to the instrument of the server
                    #off_time =int( max(0,int(self.future_events[array3]['time'] + (midi_time-self.future_events[array3]['time']) + max(0,self.future_events[array3]['service_time'])))) 
                    off_time = int( max(self.previous_time, int(self.future_events[array3]['time'] + (midi_time-self.future_events[array3]['time']) + max(0,self.future_events[array3]['service_time']))))
                    self.previous_time = off_time
                    #self.track.append(mido.Message('program_change', program=self.instruments[array3], time=off_time))
                    self.track.append(mido.Message('note_off', channel=0, note=int(self.note_offsets[array3]), velocity=int(self.future_events[array3]['velocity']),  time=off_time))

                if array3 in self.queue_lengths:
                    self.queue_lengths[array3] -= 1
                else:
                    self.queue_lengths[array3] = 0

            elif array4 == 'processing' and  ( int(array2) % self.skip_1 == 0 or int(array2) % self.skip_2 == 0 or int(array2) % self.skip_3 == 0):
                self.future_events[array3]['service_time'] += midi_time

    def save_midi(self, filename):

        # remove midi messages beyond a certain time
        for msg in self.track:
            if msg.time > 200:
                self.track.remove(msg)


        # add the end of track message
        self.track.append(mido.MetaMessage('end_of_track'))


        self.clean_midi_file()

        # add the track to the midi file
        self.mid.tracks.append(self.track)
        # save the midi file
        self.mid.save(filename)
        print("Successfully saved midi file")


    def clean_midi_file(self):
        note_on_times = {}
        msgs_to_remove = []
        for j, msg in enumerate(self.track):
            if msg.type == 'note_on':
                if msg.note in note_on_times and note_on_times[msg.note] > 0:
                    msgs_to_remove.append(j)
                else:
                    note_on_times[msg.note] = msg.time  # update the time for this note
            elif msg.type == 'note_off':
                if msg.note not in note_on_times or note_on_times[msg.note] == 0:
                    msgs_to_remove.append(j)
                else:
                    note_on_times[msg.note] = 0
            if msg.time > 200 and j not in msgs_to_remove:
                msgs_to_remove.append(j)
        for index in sorted(msgs_to_remove, reverse=True):
            self.track.pop(index)


    def sort_midi_file(self, midi_file):
        for track in midi_file.tracks:
            # Sort messages by time
            track.sort(key=lambda msg: msg.time)

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

def process_adjsim_log(n=5000, baseline=70, range=50, instruments=np.arange(0,16), note_levels=np.random.randint(0, 127, 16), gen2_output=None, count=0, start=0, end=30):
    # Example usage:
    log_processor = LogLineProcessor(r"INFO:root:([0-9]*\.[0-9]+|[0-9]+) - ([0-9]*\.[0-9]+|[0-9]+) - ([0-9]*\.[0-9]+|[0-9]+) - (arrival|departure)")

    count = 0
    max = 5000

    midi_generator = MidiGenerator(n=max, baseline=baseline, range=range, instruments=instruments, note_levels=note_levels, gen2_output=gen2_output)

    if gen2_output is None:
        gen2_output = np.random.rand(20)

    try:
        # Read the log file line by line
        with open("./logs/simulation.log", 'r') as f:
            for line in f:
                count += 1
                if count > max:
                    break
                processed_line = log_processor.process_line(line)
                if processed_line:
                    midi_generator.process_line(processed_line)

    except:
        raise ValueError("Error in processing log file")

    try:
        if count % 200 == 0:
            # save the midi file
            midi_generator.save_midi('./adj_sim_outputs/midi/simulation.mid')
    except:
        raise ValueError("Error in saving midi file")

    return generate_piano_roll(midi_generator.mid, start=start, end=end)