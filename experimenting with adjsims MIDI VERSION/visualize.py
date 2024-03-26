import matplotlib.pyplot as plt
import mido
import pretty_midi

import seaborn as sns
import numpy as np

import glob
def generate_piano_roll(midi_input, sequence_length=300, beats_length=50):
    # Check if input is a file path or a mido.MidiFile object
    if isinstance(midi_input, str):
        midi = mido.MidiFile(midi_input)
        pretty_midi_obj = pretty_midi.PrettyMIDI(midi_input)
    elif isinstance(midi_input, mido.MidiFile):
        midi = midi_input
        pretty_midi_obj = pretty_midi.PrettyMIDI(midi.filename)
    else:
        raise ValueError("midi_input must be a file path or a mido.MidiFile object")
    # Initialize piano roll array and duration array
    piano_roll = np.zeros((128, sequence_length))
    durations = np.zeros((128, sequence_length))

    # record total time
    total_time = 0

    # Convert MIDI events to piano roll representation
    my_time = 0
    note_on_time = np.zeros(128)  # to keep track of when each note was turned on
    for msg in midi:
        my_time += msg.time
        time_step = int(round(my_time))  # convert time to nearest time step
        total_time += time_step
        if time_step >= sequence_length:
            break  # stop if the sequence length is exceeded
        if msg.type == 'note_on':
            # We use time_step as the x axis (time step) and note number as the y axis
            piano_roll[msg.note, time_step] = msg.velocity
            note_on_time[msg.note] = time_step
        elif msg.type == 'note_off':
            note_off_time = int(round(note_on_time[msg.note]))
            durations[msg.note, note_off_time:time_step] = time_step - note_off_time

    # Generate beats
    beats = pretty_midi_obj.get_beats()

    # Ensure beats is of length beats_length
    if len(beats) < beats_length:
        # If beats is too short, pad it with zeros
        beats = np.pad(beats, (0, beats_length - len(beats)))
    elif len(beats) > beats_length:
        # If beats is too long, truncate it
        beats = beats[:beats_length]

    del pretty_midi_obj
    del midi

    return piano_roll, durations, beats, total_time

sequence_length = 300
beats_length = 50

file_list = sorted(glob.glob('data\\maestro-v3.0.0\\**\\*.midi', recursive=True))

# load random midi file from file_list
midi_name = file_list[0]
piano_roll, durations, beats, _ = generate_piano_roll(midi_name, sequence_length, beats_length)

def visualize_piano_roll(piano_roll):
    plt.figure(figsize=(10, 6))
    for i in range(piano_roll.shape[0]):
        plt.plot(piano_roll[i], label=f'Note {i+1}')
    plt.title('Piano Roll')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity')
    plt.legend(loc='upper right')
    plt.show()

visualize_piano_roll(piano_roll)