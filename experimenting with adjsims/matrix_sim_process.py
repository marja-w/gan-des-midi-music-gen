from midi2audio import FluidSynth
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import time

from sim_log_process_music import process_adjsim_log
from simulation_v3 import Sim

from util import get_melspectrogram_db_tensor_from_file


def matrix_to_wav(matrices=[None], size=20, use_same_instrument=None):
    num_aug = 5
    spectrograms = []
    for index, matrix in enumerate(matrices):
        if matrix == None:
            matrix = np.random.rand(size, size)
            # zero out ends of each row
            matrix[size - num_aug:, :] = 0
            matrix[:, size - num_aug:] = 0

            # for the last 4 rows up to the 24th column, randomly set the values between 0 and 1
            matrix[size - num_aug, :size - num_aug] = np.random.rand(size - num_aug)
            matrix[size - num_aug + 1, :size - num_aug] = np.random.rand(size - num_aug)
            matrix[size - num_aug + 2, :size - num_aug] = np.random.rand(size - num_aug)
            matrix[size - num_aug + 3, :size - num_aug] = np.random.rand(size - num_aug)
            matrix[size - num_aug + 4, :size - num_aug] = np.random.rand(size - num_aug)

        matrix = matrix.squeeze().detach().numpy()  # remove channel dimension
        # select source and sink nodes based on the values in the 23rd row where the values are between 0 and 1
        sources = np.where(matrix[size - num_aug] > 0.75)
        if len(sources) == 0:
            sources = np.random.choice(size - num_aug, size=size // 8, replace=False)

        instruments = np.zeros(size - num_aug)
        # select instruments for each server based on the values in 24th row where the values are between 0 and 1 and the instrument is selected based on the value up to 128
        if use_same_instrument == None:
            for i in range(size - num_aug):
                instruments[i] = int(matrix[size - num_aug + 1, i] * 126)
        else:
            instruments = np.array([use_same_instrument] * (size - num_aug))
        # print("Instruments:", instruments)

        # create a note level for each server based on the values in the 27th row where the values are between 0 and 1 and the note level is selected based on the value up to 127
        note_levels = np.zeros(size - num_aug)
        for i in range(size - num_aug):
            note_levels[i] = int(matrix[size - num_aug + 2, i] * 126)
            # print("Note levels:", note_levels)
        # print("len(note_levels):", len(note_levels))

        # create a normal distribution for each server based on the values in the 25th and 26th rows where the values are between 0 and 1
        distributions = []
        for i in range(size - num_aug):
            # distributions.append(['exponential', 1+matrix[size-num_aug+2,i]])
            if i in sources[0]:
                distributions.append(['normal', 30 * matrix[size - num_aug + 3, i], 15 * matrix[size - num_aug + 4, i]])
            else:
                distributions.append(['normal', 5 * matrix[size - num_aug + 3, i], 3 * matrix[size - num_aug + 4, i]])
        # print("Distributions:", distributions)

        for i in sources:
            matrix[:, i] = 0
            matrix[i, i] = 0

        for i in [x for x in np.arange(0, size) if x not in sources[0]]:
            matrix[i][i] = 0

        for i in range(size - num_aug):
            matrix[i] = matrix[i] / sum(matrix[i])

        for i in sources:
            matrix[i, i] = 1.0

        for i in [x for x in np.arange(0, size - num_aug) if x not in sources[0]]:
            matrix[i][i] = -1.0

        queue_list = [127] * size
        length_mel = 0
        while length_mel < 2:
            np.random.seed(np.random.randint(0, 99999, size=1))
            seeds = np.random.randint(0, 99999, size=1)
            sim_matrix = matrix[:size - num_aug, :size - num_aug]
            sim = Sim(sim_matrix, distributions, queue_list, seeds=seeds, log_path="logs/", generate_log=True,
                      animation=False, record_history=False, logging_mode='Music')
            sim.run(number_of_customers=10000)

            file_path = process_adjsim_log(instruments=instruments, note_levels=note_levels)

            fs = FluidSynth(sound_font='FluidR3_GM.sf2', sample_rate=44100)

            output_file = 'adj_sim_outputs/wav/output_' + str(index) + '.wav'

            # check if the file path exists, if not, create the file
            if not os.path.exists(output_file):
                print('Creating wav file:', output_file)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write('')

            fs.midi_to_audio(file_path, output_file)

            print('Generated wav file:', output_file)

            time.sleep(0.2)
            # create a mel spectrogram for the wav file and save it
            mel = get_melspectrogram_db_tensor_from_file(file_path=output_file)  # TODO: fix dimensions of output file data
            length_mel = mel.shape[1]

        spectrograms.append(mel)

    # return numpy array for first 5 seconds of each spectrogram
    return spectrograms
