# created with help from copilot

import numpy as np
import mido
import torch
from torch.utils.data import Dataset, DataLoader
import glob

import pretty_midi

def generate_piano_roll(midi_input, sequence_length=100, beats_length=50):
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

    # Convert MIDI events to piano roll representation
    time = 0
    note_on_time = np.zeros(128)  # to keep track of when each note was turned on
    for msg in midi:
        time += msg.time
        time_step = int(round(time))  # convert time to nearest time step
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

    return piano_roll, durations, beats

class MaestroDataset(Dataset):
    def __init__(self, root_dir, sequence_length=100, beats_length=50):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.beats_length = beats_length
        self.file_list = sorted(glob.glob('data\\maestro-v3.0.0\\**\\*.midi', recursive=True))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        midi_name = self.file_list[idx]
        piano_roll, durations, beats = generate_piano_roll(midi_name, self.sequence_length, self.beats_length)
        piano_roll = torch.FloatTensor(piano_roll)
        durations = torch.FloatTensor(durations)
        beats = torch.FloatTensor(beats)
        return piano_roll, durations, beats
    
import unittest
class TestPianoRollGeneration(unittest.TestCase):
    def test_generate_piano_roll(self):
        # Test that generate_piano_roll returns the correct output shape
        midi_file = 'adj_sim_outputs\midi\output.mid'
        sequence_length = 100
        min_beats_length = 50
        piano_roll, durations, beats = generate_piano_roll(midi_file, sequence_length)
        self.assertEqual(piano_roll.shape, (128, sequence_length))
        self.assertEqual(durations.shape, (128, sequence_length))
        self.assertGreaterEqual(len(beats), min_beats_length)

    def test_maestro_piano_roll_dataset(self):
        # Test that MaestroPianoRollDataset returns the correct output shape
        midi_files = ['adj_sim_outputs\midi\output.mid', 'adj_sim_outputs\midi\output.mid', 'adj_sim_outputs\midi\output.mid']  # replace with paths to real MIDI files
        sequence_length = 100
        min_beats_length = 50
        dataset = MaestroDataset(midi_files, sequence_length)
        piano_roll, durations, beats = dataset[0]
        self.assertEqual(piano_roll.shape, (128, sequence_length))
        self.assertEqual(durations.shape, (128, sequence_length))
        self.assertEqual(len(beats), min_beats_length)

if __name__ == '__main__':
    unittest.main()