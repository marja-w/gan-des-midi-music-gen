# created with help from copilot

import numpy as np
import mido
import torch
from torch.utils.data import Dataset, DataLoader
import glob

import pickle

import pretty_midi

def generate_piano_roll(midi_input, sequence_length=100, beats_length=50, start=0, end=50):
    if sequence_length is None:
        sequence_length = end + 20
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

    try:
        piano_roll = np.zeros((128, end-start))
        durations = np.zeros((128, end-start))

        # Convert MIDI events to piano roll representation
        my_time = 0
        note_on_time = np.zeros(128)  # to keep track of when each note was turned on
        for msg in midi:
            my_time += msg.time
            time_step = int(round(my_time))  # convert time to nearest time step
            if time_step >= sequence_length:
                break  # stop if the sequence length is exceeded
            if msg.type == 'note_on':
                # We use time_step as the x axis (time step) and note number as the y axis
                piano_roll[msg.note, time_step] = msg.velocity
                note_on_time[msg.note] = time_step
            elif msg.type == 'note_off':
                note_off_time = int(round(note_on_time[msg.note]))
                durations[msg.note, note_off_time:time_step] = time_step - note_off_time
    except:
        print(f"Error in processing midi file {midi_input}")

    if end < len(piano_roll):
        piano_roll = piano_roll[:, start:end]
        durations = durations[:, start:end]
    else:
        piano_roll = piano_roll[:, :end]
        durations = durations[:, :end]

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

    return piano_roll, durations, beats

# USES PICKLE FILE
class MaestroDatasetPickle(Dataset):
    def __init__(self,  root_dir, sequence_length=100, beats_length=50, device='cpu'):
        self.device = device
        with open('data\\preprocessed_data.pkl', 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        piano_roll, durations, beats = self.data[idx]
        piano_roll = piano_roll.to(self.device)
        durations = durations.to(self.device)
        beats = beats.to(self.device)
        return piano_roll, durations, beats
    
# USES TORCH FILES
class MaestroDatasetTorch(Dataset):
    def __init__(self, root_dir, sequence_length=100, beats_length=50, device='cpu'):
        self.data_dir = root_dir
        self.device = device
        self.file_list = sorted(glob.glob('data\\tensors\\*.pt'))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return torch.load(self.file_list[idx])

# USES MIDI FILES
class MaestroDatasetMidi(Dataset):
    def __init__(self, root_dir, sequence_length=100, beats_length=50, device='cpu'):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.beats_length = beats_length
        self.device = device
        self.file_list = sorted(glob.glob('data\\maestro-v3.0.0\\**\\*.midi', recursive=True))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        print(f"Loading data for index {idx}")
        midi_name = self.file_list[idx]
        piano_roll, durations, beats = generate_piano_roll(midi_name, self.sequence_length, self.beats_length)
        print(f"Data loaded, converting to tensors")
        piano_roll = torch.from_numpy(piano_roll).float().to(self.device)
        durations = torch.from_numpy(durations).float().to(self.device)
        beats = torch.from_numpy(beats).float().to(self.device)
        print(f"Tensors created for index {idx}")
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
        sequence_length = 30
        min_beats_length = 50
        dataset = MaestroDatasetMidi(midi_files, sequence_length)
        piano_roll, durations, beats = dataset[0]
        self.assertEqual(piano_roll.shape, (128, sequence_length))
        self.assertEqual(durations.shape, (128, sequence_length))
        self.assertEqual(len(beats), min_beats_length)

if __name__ == '__main__':
    unittest.main()