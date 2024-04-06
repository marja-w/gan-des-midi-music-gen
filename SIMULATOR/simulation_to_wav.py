import numpy as np
from midi2audio import FluidSynth
import os

from simulation_v3 import Sim

def matrix_to_wav(matrices=[None], size=32, use_same_instrument=None, sound_font='FluidR3_GM.sf2'):
    num_aug = 5
    
    for index, matrix in enumerate(matrices):
        if matrix is None:
            matrix = np.random.rand(size,size)
            # zero out ends of each row
            matrix[size-num_aug:,:] = 0
            matrix[:,size-num_aug:] = 0
            
            # for the last 4 rows up to the 24th column, randomly set the values between 0 and 1
            matrix[size-num_aug, :size-num_aug] = np.random.rand(size-num_aug)
            matrix[size-num_aug+1, :size-num_aug] = np.random.rand(size-num_aug)
            matrix[size-num_aug+2, :size-num_aug] = np.random.rand(size-num_aug)
            matrix[size-num_aug+3, :size-num_aug] = np.random.rand(size-num_aug)
            matrix[size-num_aug+4, :size-num_aug] = np.random.rand(size-num_aug)

        # select source and sink nodes based on the values in the 23rd row where the values are between 0 and 1
        sources = np.where(matrix[size-num_aug] > 0.75)
        if len(sources) == 0:
            sources = np.random.choice(size-num_aug, size=size//8, replace=False)


        instruments = np.zeros(size-num_aug)
        # select instruments for each server based on the values in 24th row where the values are between 0 and 1 and the instrument is selected based on the value up to 128
        if use_same_instrument == None:
            for i in range(size-num_aug):
                instruments[i] = int(matrix[size-num_aug+1,i] * 127)
        else:
            instruments = np.array([use_same_instrument]*(size-num_aug))
        #print("Instruments:", instruments)


        # create a note level for each server based on the values in the 27th row where the values are between 0 and 1 and the note level is selected based on the value up to 127
        note_levels = np.zeros(size-num_aug)
        for i in range(size-num_aug):
            note_levels[i] = int(matrix[size-num_aug+2,i] * 127) 
        #print("Note levels:", note_levels)
        #print("len(note_levels):", len(note_levels))

        # create a normal distribution for each server based on the values in the 25th and 26th rows where the values are between 0 and 1
        distributions = []
        for i in range(size-num_aug):
            #distributions.append(['exponential', 1+matrix[size-num_aug+2,i]])
            if i in sources[0]:
                distributions.append(['normal', 10*matrix[size-num_aug+3,i], 5*matrix[size-num_aug+4,i]])
            else:
                distributions.append(['normal', 3*matrix[size-num_aug+3,i], 2*matrix[size-num_aug+4,i]])
        #print("Distributions:", distributions)

        for i in sources:
            matrix[:,i] = 0
            matrix[i,i] = 0

        for i in [x for x in np.arange(0,size) if x not in sources[0]]:
            matrix[i][i] = 0

        for i in range(size-num_aug):
            matrix[i] = matrix[i] / sum(matrix[i])

        for i in sources:
            matrix[i,i] = 1.0

        for i in [x for x in np.arange(0,size-num_aug) if x not in sources[0]]:
            matrix[i][i] = -1.0

        queue_list = [127] * size


        np.random.seed(np.random.randint(0, 99999, size=1))
        seeds = np.random.randint(0, 99999, size=1)
        sim_matrix = matrix[:size-num_aug, :size-num_aug]
        sim = Sim(sim_matrix, distributions, queue_list, seeds=seeds, generate_log=True, animation=False, record_history=False, logging_mode='Music')
        sim.run(number_of_customers=1000)

        file_path = process_adjsim_log(instruments=instruments, note_levels=note_levels)

        fs = FluidSynth(sound_font=sound_font, sample_rate=44100)

        output_file = 'adj_sim_outputs\wav\output_'+ str(index) + '.wav'

        # check if the file path exists, if not, create the file
        if not os.path.exists(output_file):
            print('Creating wav file:', output_file)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write('')

        fs.midi_to_audio(file_path, output_file)

        print('Generated wav file:', output_file)

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
        if note_levels != []:
            for i,note_level in enumerate(note_levels):
                self.note_offsets[str(i)] = int(note_level)
        else:
            for i in range(0,32):
                self.note_offsets[str(i)] = random.randint(self.baseline-self.range, self.baseline+self.range)

        self.queue_lengths = {}

        self.instruments = {}
        if instruments != []:
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
        self.track.append(mido.MetaMessage('set_tempo', tempo=1000000, time=0))

        # set the time signature
        self.track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        # set the key signature
        self.track.append(mido.MetaMessage('key_signature', key='C', time=0))

        # set the instrument
        self.track.append(mido.Message('program_change', program=0, time=0))


    def process_line(self, processed_line):
        array1, array2, array3, array4 = processed_line
        # time, event, server, arrival/departure

        # calculate midi time based on value in array1
        midi_time = max(0,int(float(array1)))

        if array4 == 'arrival' and  ( int(array2) % 3 == 0 or int(array2) % 5 == 0 or int(array2) % 7 == 0):
            if array3 in self.queue_lengths:
                self.queue_lengths[array3] += 1
            else:
                self.queue_lengths[array3] = 1

            queue_length = self.queue_lengths[array3]
            if 127 <= queue_length < 2*127:
                queue_length = min(127,max(0, 2*127 - queue_length))
            elif queue_length >= 2*127:
                queue_length = min(127,max(0, queue_length % 127))

            max_customer_id = max(1,(30 + queue_length) % 127)
            customer_id = int(array2)
            if max_customer_id <= customer_id < 2*max_customer_id:
                customer_id = min(max_customer_id,max(0, 2*max_customer_id - customer_id))
            elif customer_id >= 2*max_customer_id:
                customer_id = min(max_customer_id,max(0, customer_id % max_customer_id))

            self.future_events[array3] = {}
            self.future_events[array3]['time'] = midi_time
            self.future_events[array3]['velocity'] = 60 + (int(customer_id) % 67)
            self.future_events[array3]['service_time'] = int(queue_length)


        elif array4 == 'departure' and  ( int(array2) % 3 == 0 or int(array2) % 5 == 0 or int(array2) % 7 == 0):

            if array3 in self.future_events:
                # change the instrument to the instrument of the server
                on_time = max(0, int(self.future_events[array3]['time']))
                self.track.append(mido.Message('program_change', program=self.instruments[array3], time=on_time))
                self.track.append(mido.Message('note_on', channel=0, note=self.note_offsets[array3], velocity=int(self.future_events[array3]['velocity']),  time=on_time))

                # change the instrument to the instrument of the server
                off_time = max(0,int(self.future_events[array3]['time'] + (midi_time-self.future_events[array3]['time']) + max(0,self.future_events[array3]['service_time']))) 
                self.track.append(mido.Message('program_change', program=self.instruments[array3], time=off_time))
                self.track.append(mido.Message('note_off', channel=0, note=self.note_offsets[array3], velocity=self.future_events[array3]['velocity'],  time=off_time))


            if array3 in self.queue_lengths:
                self.queue_lengths[array3] -= 1
            else:
                self.queue_lengths[array3] = 0

        elif array4 == 'processing' and  ( int(array2) % 3 == 0 or int(array2) % 5 == 0 or int(array2) % 7 == 0):
            self.future_events[array3]['service_time'] += midi_time


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