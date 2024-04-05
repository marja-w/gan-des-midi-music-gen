import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from util import get_melspectrogram_db_tensor, get_melspectrogram_db_tensor_from_file, split_audio_data
from util import get_melspectrogram_db_tensor_maestro
from midi2audio import FluidSynth


class InputSong(Dataset):
    """
    PyTorch Dataset for loading one song. It cuts the song in excerpts according to window_size and
    hop_length_audio parameters of the __init__() function
    """

    def __init__(self, audio_file, window_size=5,
                 hop_length_audio=5):  # TODO: write function in utils for cutting audio
        # get audio parameters
        waveform, sample_rate = torchaudio.load(audio_file, normalize=True)
        self.orig_waveform = waveform
        self.sample_rate = sample_rate

        # self.audio_file_length = len(waveform[1]) / self.sample_rate
        self.audio_file_length = waveform.size(dim=1) / sample_rate

        # split audio TODO: put in utils
        self.window_size = window_size  # length in seconds
        self.hop_length_audio = hop_length_audio  # window stride in seconds
        self.audio_files = list()
        channel = 0  # TODO what channel to use
        for i in np.arange(0, len(waveform[channel]) + 1, hop_length_audio * sample_rate):
            if i + hop_length_audio * sample_rate > len(waveform[channel]):
                # make sure last sample is as long as the others
                self.audio_files.append(waveform[channel][-hop_length_audio * sample_rate:])
            else:
                self.audio_files.append(waveform[channel][i:i + hop_length_audio * sample_rate])

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, item):
        wav = self.audio_files[item]
        # Compute spectrogram
        spectrogram = get_melspectrogram_db_tensor(wav, self.sample_rate)  # [(110250,), 22050]
        return spectrogram  # (128, 216)


class MaestroDataset(Dataset):
    def __init__(self, batch_size):
        """
        Get index to filepath mapping from metadata files of the dataset
        """
        self.INPUT_FOLDER = "../data/maestro-v3.0.0"
        self.meta_data_file = f"{self.INPUT_FOLDER}/maestro-v3.0.0.json"
        self.OUTPUT_PATH = "./data/maestro.wav"  # temporarily store the .wav file here
        self.k = batch_size

        with open(self.meta_data_file) as json_file:
            data = json.load(json_file)
            self.data = data['midi_filename']

        # store FluidSynth object for midi to wav conversion
        self.fs = FluidSynth(sound_font='FluidR3_GM.sf2', sample_rate=44100)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        """
        Get a midi file from the index and create a .wav file from it
        :param index: the index of the midi file to access
        :return: the audio data
        """
        input_midi_file_path = f"{self.INPUT_FOLDER}/{self.data[str(index)]}"
        self.fs.midi_to_audio(input_midi_file_path, self.OUTPUT_PATH)
        splits = split_audio_data(self.OUTPUT_PATH)
        mels = list()
        if len(splits) > self.k:
            splits = random.sample(splits, self.k)
        for split in splits:
            mel = get_melspectrogram_db_tensor(split)
            mels.append(mel)
        output = torch.stack(mels)
        return output


def my_collate(batch):
    """
    receive a list of samples returned by Dataset.__getitem__ and creates the final batch
    :param batch:
    :return: batch of size (x, 128, 431) where x is the number of splits for all songs in batch
    """
    return torch.cat(batch)  # stack the tensors with different number of rows x

if __name__ == '__main__':
    input_song = InputSong(audio_file="../data/classical.00000.wav")
    maestro_dataset = MaestroDataset(batch_size=30)

    dataloader_input_song = DataLoader(input_song, batch_size=2, shuffle=True)
    dataloader_maestro = DataLoader(maestro_dataset, batch_size=1, shuffle=True, collate_fn=my_collate)

    for data in dataloader_input_song:
        print(data.shape)