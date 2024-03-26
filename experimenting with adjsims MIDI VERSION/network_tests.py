# created with help from copilot... although at times it was not very helpful


import datetime

import numpy as np
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import dataloader, Dataset
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from util import get_melspectrogram_db_from_file, get_melspectrogram_db_tensor
import scipy.io.wavfile as wav

from datasets import MaestroDatasetPickle, MaestroDatasetMidi, MaestroDatasetTorch, generate_piano_roll

from torch.optim.lr_scheduler import StepLR

import unittest
import glob

from matrix_sim_process import matrix_to_midi

import time


def get_noise(n_samples, noise_dim, device='cpu'):
    return torch.randn(n_samples, noise_dim, device=device)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean=0.5, std=0.1)
        torch.nn.init.constant_(m.bias, val=0)
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.5, std=0.1)
        torch.nn.init.constant_(m.bias, val=0)



class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64, adj_size=None, device='cpu'):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.adj_size = adj_size  # store adj_size
        self.device = device
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan * adj_size[0] * adj_size[1]),
        )
        self.gen.apply(weights_init)

    def make_gen_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        gen_output = self.gen(noise)
        gen_output = gen_output.view(len(noise), -1, self.adj_size[0], self.adj_size[1])
        return gen_output.to(self.device)


class BeatGenerator(nn.Module):
    def __init__(self, z_dim=10, hidden_dim=64,  input_dim=None, output_dim=None, device='cpu'):
        super(BeatGenerator, self).__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim  # store output_dim
        self.input_tensor_dim = input_dim
        self.device = device
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim + input_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, output_dim),
        )
        self.gen.apply(weights_init)

    def make_gen_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise, input_tensor=None):
        # generate random tensor of length input_densor_dim if input_tensor is None
        if input_tensor == None:
            input_tensor = torch.randn(len(noise), self.input_tensor_dim).to(self.device)

        combined_input = torch.cat((noise, input_tensor), dim=1).to(self.device)
        return self.gen(combined_input)


class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=16, roll_size=None, device='cpu'):
        super(Discriminator, self).__init__()
        self.roll_size = roll_size  # store adj_size
        self.device=device
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan * roll_size[0] * roll_size[1] * roll_size[2], hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1),
        )

    def make_disc_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image):
        return self.disc(image)
            

class DiscriminatorCNN(nn.Module):
    def __init__(self, roll_size=(2, 128, 30), hidden_dim=16):
        super(DiscriminatorCNN, self).__init__()
        self.conv1 = nn.Conv2d(roll_size[0], hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.final_size = hidden_dim * 2 * ((roll_size[1] // 4) * (roll_size[2] // 4))
        self.fc = nn.Linear(self.final_size, 1)

    def forward(self, image):
        x = self.leaky_relu(self.conv1(image))
        x = self.leaky_relu(self.conv2(x))
        x = x.view(len(x), -1)  # Flatten the tensor
        return self.fc(x)


class MultiModalGAN(nn.Module):
    def __init__(self, z_dim=100, hidden_dim=64, adj_size=(28, 28),roll_size=(2,128,100), input_dim=50, output_dim=16, instrument=None, start=30, end=60, device='cpu'):
        super(MultiModalGAN, self).__init__()
        self.z_dim = z_dim
        self.generator1 = Generator(z_dim, hidden_dim=hidden_dim, adj_size=adj_size, device=device).to(device)
        self.generator2 = BeatGenerator(z_dim, hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim, device=device).to(device)
        self.discriminator = DiscriminatorCNN(roll_size=roll_size).to(device)
        self.instrument = instrument
        self.start = start
        self.end = end
        self.adj_size = adj_size 
        self.device = device

    def forward(self, noise1, noise2, input_tensor, count):
        gen_output1 = self.generator1(noise1)
        gen_output2 = self.generator2(noise2, input_tensor)

        start = time.time()
        sim_output, failed_sim_count = matrix_to_midi(gen_output1, gen_output2, adj_size=self.adj_size, instrument=self.instrument, start=self.start, end=self.end, count=count)
        print("matrix_to_midi took", time.time() - start, "seconds")

        # MAYBE NEEDS TO BE FIXED... WILL TEST ON NEXT ITERATION WITH NEW DATA ( IF NO ERROR, CAN BE REMOVED )
        sim_output = [torch.from_numpy(batch).float().to(self.device) for batch in sim_output]
        sim_output = torch.stack(sim_output)

        disc_output = self.discriminator(sim_output)

        return disc_output, failed_sim_count

class TestMultiModalGAN(unittest.TestCase):
    def test_training_loop(self, batch_size=16):

        gen2_output_dim = 20 # output dimension of the second generator for simulator parameters
        max_beat_length = 50 # maximum length of the beat vector
        noise_dim = 50 # dimension of the noise vector
        adj_size = (32,32) # dimensions of the adjacency matrix
        
        start = 30 # start of the sequence
        sequence_length = 30 # length of the sequence generated by simulator   NOTE: TO CHANGE THIS, YOU NEED TO RE-PICKLE THE DATASET WITH THE NEW SEQUENCE LENGTH
        
        roll_size = (2,128,sequence_length) # dimensions of the piano roll matrix First channel is note velocity, second channel is time step


        # get the correct device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Starting dataset creation...")
        start = time.time()
        dataset = MaestroDatasetPickle('preprocessed_data.pkl', beats_length=max_beat_length, sequence_length=sequence_length, device=device)
        train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
        print("Dataset creation took", time.time() - start, "seconds")  

        print("Starting setup of model...")
        start = time.time()
        mmgan = MultiModalGAN(z_dim=noise_dim, adj_size=adj_size, roll_size=roll_size, input_dim=max_beat_length, output_dim=gen2_output_dim, instrument=0, start=start, end=start+sequence_length, device=device)
        criterion = nn.BCEWithLogitsLoss()
        gen_opt = torch.optim.Adam(mmgan.parameters(), lr=0.01)
        scheduler = StepLR(gen_opt, step_size=30, gamma=0.1)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_opt, 'min', patience=10)
        print("Model setup took", time.time() - start, "seconds")

        num_epochs = 30
        print_interval = 10
        save_interval = 5  # Save the model every 5 epochs

        count = 0

        total_failures = 0
        total_seen = 0

        for epoch in range(num_epochs):
            mmgan.train()
            train_losses = []
            for i, (piano_roll, durations, beats) in enumerate(train_loader):
                print("Batch", count)
                count += 1
                noise1 = torch.randn(batch_size, noise_dim, device=device)
                noise2 = torch.randn(batch_size, noise_dim, device=device)
                real = torch.ones(batch_size, 1, device=device).view(-1)
                fake, failed_sim_count = mmgan(noise1, noise2, beats, count)
                fake = fake.squeeze(1)
                gen_loss = criterion(fake, real)
                gen_loss.backward()
                gen_opt.step()
                gen_opt.zero_grad()
                scheduler.step()

                """
                start = time.time()
                print("Updating weights...")
                gen_opt.zero_grad()
                gen_loss = criterion(fake, real) + failed_sim_count/batch_size
                gen_loss.backward()
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(mmgan.generator1.parameters(), max_norm=1)
                torch.nn.utils.clip_grad_norm_(mmgan.generator2.parameters(), max_norm=1)
                gen_opt.step()
                scheduler.step(gen_loss)
                #scheduler.step()
                print("Updating weights took", time.time() - start, "seconds")
                """
                total_failures += failed_sim_count
                total_seen += batch_size

                train_losses.append(gen_loss.item())
                if (i + 1) % print_interval == 0:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Step {i + 1}/{len(train_loader)}, Train Loss: {gen_loss.item()}')

            print(f'Epoch {epoch + 1}/{num_epochs}, Avg Train Loss: {sum(train_losses) / len(train_losses)}')
            print("Total failures:", total_failures, "Total seen:", total_seen)

            # timeout for 10 seconds
            time.sleep(30)

            # Save the model every save_interval epochs
            if (epoch + 1) % save_interval == 0:
                torch.save(mmgan.state_dict(), f'models//mmgan_epoch_{epoch + 1}.pth')

        return train_losses
    
if __name__ == '__main__':
    unittest.main()