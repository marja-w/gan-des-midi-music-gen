from midi2audio import FluidSynth
import numpy as np
import matplotlib.pyplot as plt

from sim_log_to_midi import process_adjsim_log
from simulation_v3 import Sim

import threading

import time

def run_simulation(sim, num_customers):
    sim.run(number_of_customers=num_customers)

def matrix_to_midi(gen1_output, gen2_output, adj_size=(32,32), instrument=None, start=0, end=150, count=0, generate=False):
    num_aug = 3
    midi_rolls = []

    debug_print = False

    start = int(start)
    end = int(end)

    size = adj_size[0]

    dim = size - num_aug

    gen1_output = gen1_output.cpu().detach().numpy()
    gen2_output = gen2_output.cpu().detach().numpy()

    failed_simulations = 0

    for index, matrix in enumerate(gen1_output):

        matrix = matrix[0]
        matrix = np.abs(matrix)

        #print("Matrix:", matrix)

        # select source and sink nodes based on the values in the 23rd row where the values are between 0 and 1
        sources = np.where(matrix[dim] > gen2_output[index][0])
        #sources = np.where(matrix[dim] > 0.5 )
        if len(sources[0]) == 0 or len(sources[0] == dim):
            sources = np.random.choice(dim, size=(dim) // 4, replace=False)
        else:
            sources = sources[0]

        servers = [x for x in np.arange(0, size-num_aug) if x not in sources]

        #print("Sources:", sources)
        #print("Servers:", servers)

        instruments = np.zeros(dim)
        # select instruments for each server based on the values in 24th row where the values are between 0 and 1 and the instrument is selected based on the value up to 128
        if instrument == None:
            for i in range(dim):
                instruments[i] = int(matrix[dim + 1, i] * 126)
        else:
            instruments = np.array([instrument] * (dim))
        # print("Instruments:", instruments)

        # create a note level for each server based on the values in the 27th row where the values are between 0 and 1 and the note level is selected based on the value up to 127
        note_levels = np.ones(dim)
        for i in range(dim):
            note_levels[i] = max(0,(int(matrix[dim + 2, i] * 126)) % 128)

        # create a normal distribution for each server based on the values in the 25th and 26th rows where the values are between 0 and 1
        distributions = []
        for i in range(dim):
            # distributions.append(['exponential', 1+matrix[size-num_aug+2,i]])
            if i in sources:
                distributions.append(['normal', np.abs(gen2_output[index][1] * 50), np.abs( gen2_output[index][2] * 50) ])
            else:
                distributions.append(['normal', np.abs(gen2_output[index][3] * 10), np.abs(gen2_output[index][4] * 10) ])
        # print("Distributions:", distributions)


        sim_matrix = matrix[:dim, :dim]

        for i in sources:
            sim_matrix[:, i] = 0.0
            sim_matrix[i, i] = 0.0

        for i in servers:
            sim_matrix[i][i] = 0.0


        # Convert to float64 for higher precision
        sim_matrix = sim_matrix.astype(np.float64)

        # Add a small constant to the sum of each row to ensure it's never 0
        row_sums = sim_matrix.sum(axis=1, keepdims=True)

        # Normalize the matrix within float32 range
        sim_matrix = sim_matrix / row_sums

        # Handle the case where row sum is 0
        sim_matrix[np.isnan(sim_matrix)] = 0

        # add difference between the sum of row and 1 to some random element in the row other than the diagonal element
        for i in range(dim):
            sim_matrix[i, np.random.choice([x for x in range(dim) if x != i and sim_matrix[i,x] != 0]) ] += 1 - sim_matrix[i].sum()

        

        for i in sources:
            sim_matrix[i,i] = 1.0

        for i in servers:
            sim_matrix[i,i] = -1.0

        queue_list = [2*127] * dim

        np.random.seed(np.random.randint(0, 99999, size=1))
        seeds = np.random.randint(0, 99999, size=1)

        num_customers = max(1000,int(3000*gen2_output[index][6]))

        this_count = 1
        if index == 0:
            this_count = count

        if this_count % 100 == 0 and debug_print:
            print("Generated", count, "simulations")
            print("Sources:", sources)
            print("Servers:", servers)
            print("Instruments:", instruments)
            print("Note Levels:", note_levels)
            print("Distributions:", distributions)
            print("Num Customers:", num_customers)
            print("Max Sim Time:", min(float(gen2_output[index][5]),1.0))
            print("Seeds:", seeds)
            print("Sim Matrix:", sim_matrix)
            print("Queue List:", queue_list)
            print("Count:", count)
            start_time = time.time()
            for i in range(dim):
                print("Row", i, "diag", sim_matrix[i, i], "sum", sim_matrix[i].sum())


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
        if num_customers < 200:
            num_customers = 200
        start_time = time.time()
        try:
            sim_thread = threading.Thread(target=run_simulation, args=(sim, num_customers))

            sim_thread.start()

            sim_thread.join(timeout=2.5)

            if sim_thread.is_alive():
                print("Simulation took too long, stopping")
                failed_simulations +=1
                roll = np.zeros((128, end - start))
                durations = np.zeros((128, end - start))
            else:
                roll, durations, _ = process_adjsim_log(instruments=instruments, note_levels=note_levels, gen2_output=gen2_output[index][10:], count=this_count, start=start, end=end, generate=generate)

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

        if this_count % 1000 == 0 and debug_print:
            print("Time taken for simulation", time.time() - start_time)
            

        del sim

        midi_rolls.append(output)


    # return numpy array for first 5 seconds of each spectrogram
    return midi_rolls, failed_simulations

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def plot_input_matrix(matrix_copy,sources,servers, ax=None):
    n = matrix_copy.shape[0]
    for i in range(n):
        matrix_copy[i, i] = np.nan

    # Create a color map suitable for probability data
    cmap = plt.get_cmap('viridis')

    # Create the graph
    im = plt.imshow(matrix_copy, cmap=cmap, vmin=np.nanmin(matrix_copy), vmax=np.nanmax(matrix_copy))

    # Overlay a scatter plot on the heatmap to change the color of the diagonal elements
    for i in range(n):
        if i in sources:
            plt.scatter(i, i, color='green', s=40)  # Green for sources
        if i in servers:
            plt.scatter(i, i, color='red', s=40)  # Red for servers


    # Create custom legend entries
    red_line = mlines.Line2D([], [], color='red', marker='o', markersize=10, label='Server', linestyle='None')
    green_line = mlines.Line2D([], [], color='green', marker='o', markersize=10, label='Source', linestyle='None')


    # Add the legend to the plot
    plt.legend(handles=[red_line, green_line])

    # Add a title
    plt.title('Matrix Transition Probabilities')

    # Set x-axis ticks at every integer value
    plt.xticks(np.arange(n))

    # Set y-axis ticks at every integer value
    plt.yticks(np.arange(n))

    # Add labels to the x and y axes
    plt.xlabel('Transition probability from row to column')

    # Add a colorbar indicating the scale of the heatmap values
    plt.colorbar(im, label='Matrix Values', cmap=cmap)

    # Display the graph
    plt.show()