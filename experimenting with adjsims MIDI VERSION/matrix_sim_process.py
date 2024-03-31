from midi2audio import FluidSynth
import numpy as np
import matplotlib.pyplot as plt

from sim_log_to_midi import process_adjsim_log
from simulation_v3 import Sim

import threading

import time

def run_simulation(sim, num_customers):
    sim.run(number_of_customers=num_customers)

def matrix_to_midi(gen1_output, gen2_output, adj_size=(32,32), instrument=None, start=0, end=150, count=0):
    num_aug = 3
    midi_rolls = []

    start = int(start)
    end = int(end)

    size = adj_size[0]

    gen1_output = gen1_output.cpu().detach().numpy()
    gen2_output = gen2_output.cpu().detach().numpy()

    failed_simulations = 0

    for index, matrix in enumerate(gen1_output):

        matrix = matrix[0]
        matrix = np.abs(matrix)

        # select source and sink nodes based on the values in the 23rd row where the values are between 0 and 1
        sources = np.where(matrix[size - num_aug] > gen2_output[index][0])
        #sources = np.where(matrix[size - num_aug] > 0.5 )
        if len(sources[0]) == 0 or len(sources[0] == size - num_aug):
            sources = np.random.choice(size - num_aug, size=(size - num_aug) // 4, replace=False)
        else:
            sources = sources[0]

        instruments = np.zeros(size - num_aug)
        # select instruments for each server based on the values in 24th row where the values are between 0 and 1 and the instrument is selected based on the value up to 128
        if instrument == None:
            for i in range(size - num_aug):
                instruments[i] = int(matrix[size - num_aug + 1, i] * 126)
        else:
            instruments = np.array([instrument] * (size - num_aug))
        # print("Instruments:", instruments)

        # create a note level for each server based on the values in the 27th row where the values are between 0 and 1 and the note level is selected based on the value up to 127
        note_levels = np.ones(size - num_aug)
        for i in range(size - num_aug):
            note_levels[i] = max(0,(int(matrix[size - num_aug + 2, i] * 126)) % 128)

        # create a normal distribution for each server based on the values in the 25th and 26th rows where the values are between 0 and 1
        distributions = []
        for i in range(size - num_aug):
            # distributions.append(['exponential', 1+matrix[size-num_aug+2,i]])
            if i in sources:
                distributions.append(['normal', np.abs(gen2_output[index][1] * 50), np.abs( gen2_output[index][2] * 50) ])
            else:
                distributions.append(['normal', np.abs(gen2_output[index][3] * 10), np.abs(gen2_output[index][4] * 10) ])
        # print("Distributions:", distributions)

        for i in sources:
            matrix[:, i] = 0
            matrix[i, i] = 0

        for i in [x for x in np.arange(0, size) if x not in sources]:
            matrix[i][i] = 0

        sucess = True
        for i in range(size - num_aug):
            try:
                matrix[i] = np.abs(matrix[i]) / np.abs(matrix[i].sum())
            except:
                # set random values to index of servers with sum to 1
                matrix[i] = np.random.rand(size - num_aug)
                matrix[i] = np.abs(matrix[i]) / np.abs(matrix[i].sum())
            if matrix[i].sum() == 0:
                sucess = False

        if not sucess:
            midi_rolls.append(np.zeros((2,128, end-start)))
            continue

        for i in sources:
            matrix[i, i] = 1.0

        for i in [x for x in np.arange(0, size - num_aug) if x not in sources]:
            matrix[i][i] = -1.0

        queue_list = [127] * size

        np.random.seed(np.random.randint(0, 99999, size=1))
        seeds = np.random.randint(0, 99999, size=1)
        sim_matrix = matrix[:size - num_aug, :size - num_aug]

        num_customers = max(1000,int(3000*gen2_output[index][6]))

        """
        print(sim_matrix.shape, len(distributions), len(queue_list), seeds, gen2_output[index][5], gen2_output[index][6])
        print("dist", distributions)
        print("queue", queue_list)
        print("num_customers", num_customers)
        print("max_sim_time", max(float(gen2_output[index][5]),1.0))
        print("-----------------")
        print(sim_matrix)
        """

        sim = Sim(sim_matrix, distributions, queue_list, seeds=seeds, log_path="logs/", generate_log=True,
                    animation=False, record_history=False, logging_mode='Music', max_sim_time=min(float(gen2_output[index][5]),1.0))
        
        output = np.zeros((2, 128,  end - start))
        if num_customers == 0:
            num_customers = 200
        start_time = time.time()
        try:
            sim_thread = threading.Thread(target=run_simulation, args=(sim, num_customers))

            sim_thread.start()

            sim_thread.join(timeout=2.5)

            if sim_thread.is_alive():
                print("Simulation took too long, stopping")
                failed_simulations +=1
            else:
                roll, durations, _ = process_adjsim_log(instruments=instruments, note_levels=note_levels, gen2_output=gen2_output[index][10:], count=count, start=start, end=end)

                if roll is None:
                    failed_simulations += 1
                    midi_rolls.append(output)
                    continue

                output[0] = roll
                output[1] = durations
        except:
            print("Error in simulation thread, using blank piano roll instead.")
            failed_simulations += 1
            raise ValueError("Error in simulation thread, using blank piano roll instead.")
            

        del sim

        midi_rolls.append(output)


    # return numpy array for first 5 seconds of each spectrogram
    return midi_rolls, failed_simulations
