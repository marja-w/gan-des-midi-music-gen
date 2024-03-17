from midi2audio import FluidSynth
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa

from sim_log_process_music import process_adjsim_log
from simulation_v3 import Sim


def get_melspectrogram_db(file_path, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    y, sr = librosa.load(file_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel

def matrix_to_wav(matrices=[None]*16, size=28):
    num_aug = 5
    for index, matrix in enumerate(matrices):
        if not matrix:
            matrix = np.random.rand(size,size)
            # zero out ends of matrix such that only the top 24x24 is non-zero
            matrix[size-4:size, :] = 0

            # for the last 4 rows up to the 24th column, randomly set the values between 0 and 1
            matrix[size-num_aug, :size-num_aug] = np.random.rand(size-num_aug)
            matrix[size-num_aug+1, :size-num_aug] = np.random.rand(size-num_aug)
            matrix[size-num_aug+2, :size-num_aug] = np.random.rand(size-num_aug)
            matrix[size-num_aug+3, :size-num_aug] = np.random.rand(size-num_aug)
            matrix[size-num_aug+4, :size-num_aug] = np.random.rand(size-num_aug)

        # select source and sink nodes based on the values in the 23rd row where the values are between 0 and 1
        sources = np.where(matrix[size-num_aug] > 0.85)

        instruments = np.zeros(size-num_aug)
        # select instruments for each server based on the values in 24th row where the values are between 0 and 1 and the instrument is selected based on the value up to 128
        for i in range(size-num_aug):
            instruments[i] = int(matrix[size-num_aug+1,i] * 127)

        # create a normal distribution for each server based on the values in the 25th and 26th rows where the values are between 0 and 1
        distributions = []
        for i in range(size-num_aug):
            distributions.append(['normal', matrix[size-num_aug+2,i], matrix[size-num_aug+3,i]])

        # create a note level for each server based on the values in the 27th row where the values are between 0 and 1 and the note level is selected based on the value up to 127
        note_levels = np.zeros(size-num_aug+4)
        for i in range(size-num_aug+4):
            note_levels[i] = int(matrix[size-num_aug+4,i] * 127)

        matrix[:, size-num_aug:size] = 0

        for i in sources:
            matrix[:,i] = 0
            matrix[i,i] = 0

        for i in [x for x in np.arange(0,size) if x not in sources[0]]:
            matrix[i][i] = 0

        for i in range(size):
            matrix[i] = matrix[i] / sum(matrix[i])

        for i in sources:
            matrix[i,i] = 1.0

        for i in [x for x in np.arange(0,size) if x not in sources[0]]:
            matrix[i][i] = -1.0

        queue_list = [127] * size

        np.random.seed(42)
        seeds = np.random.randint(0, 99999, size=1)
        sim_matrix = matrix[:size-num_aug, :size-num_aug]
        sim = Sim(sim_matrix, distributions, queue_list, seeds=seeds, generate_log=True, animation=False, record_history=False, logging_mode='Music')
        sim.run(number_of_customers=1000)

        file_path = process_adjsim_log(instruments=instruments, note_levels=note_levels)

        fs = FluidSynth(sound_font='FluidR3_GM.sf2')

        output_file = 'adj_sim_outputs\wav\output_'+ str(index) + '.wav'

        # check if the file path exists, if not, create the file
        if not os.path.exists(output_file):
            print('Creating wav file:', output_file)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write('')

        fs.midi_to_audio(file_path, output_file)

        print('Generated wav file:', output_file)

        # create a mel spectrogram for the wav file and save it
        mel = get_melspectrogram_db(file_path=output_file)
        plt.imsave('adj_sim_outputs\spectrograms\output_'+ str(index) + '.png', mel)
        print('Generated spectrogram:', 'adj_sim_outputs\spectrograms\output_'+ str(index) + '.png')

    # return numpy array for first 5 seconds of each spectrogram
    return np.array([get_melspectrogram_db(file_path='adj_sim_outputs\spectrograms\output_'+ str(index) + '.png')[:, :216] for index in range(16)])

