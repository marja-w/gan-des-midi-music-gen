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

import pickle

import os

from util import get_melspectrogram_db_from_file, get_melspectrogram_db_tensor
import scipy.io.wavfile as wav

from datasets import MaestroDatasetPickle, MaestroDatasetMidi, MaestroDatasetTorch, generate_piano_roll

from torch.optim.lr_scheduler import StepLR

import unittest
import glob

from matrix_sim_process import matrix_to_midi

import time

from torchviz import make_dot
import torchviz


def get_noise(n_samples, noise_dim, device='cpu'):
    return torch.randn(n_samples, noise_dim, device=device)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0, std=1)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.)


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64,input_dim=None, adj_size=None, device='cpu'):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.adj_size = adj_size  # store adj_size
        self.device = device
        if input_dim == None:
            input_dim = z_dim
        self.input_tensor_dim = input_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim + self.input_tensor_dim, hidden_dim * 4),
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

    def forward(self, noise, input_tensor=None):
        if input_tensor == None:
            input_tensor = torch.randn(len(noise), self.input_tensor_dim).to(self.device)

        combined_input = torch.cat((noise, input_tensor), dim=1).to(self.device)

        gen_output = self.gen(combined_input)
        gen_output = gen_output.view(len(noise), -1, self.adj_size[0], self.adj_size[1])
        return gen_output.to(self.device)


class BeatGenerator(nn.Module):
    def __init__(self, z_dim=10, hidden_dim=64,  input_dim=None, output_dim=None, device='cpu'):
        super(BeatGenerator, self).__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim  # store output_dim
        if input_dim == None:
            input_dim = z_dim
        self.input_tensor_dim = input_dim
        self.device = device
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim + self.input_tensor_dim, hidden_dim * 4),
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
    def __init__(self, z_dim=100, hidden_dim=64, adj_size=(28, 28),roll_size=(2,128,50), input_dim=50, output_dim=16, instrument=None, start=30, end=80, device='cpu'):
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
        sim_output, failed_sim_count = matrix_to_midi(gen_output1.detach(), gen_output2.detach(), adj_size=self.adj_size, instrument=self.instrument, start=self.start, end=self.end, count=count)

        # Convert simulated output to tensors
        sim_output = [torch.from_numpy(batch).float().to(self.device) for batch in sim_output]
        sim_output = torch.stack(sim_output)


        return self.discriminator(sim_output),failed_sim_count

class TestMultiModalGAN(unittest.TestCase):
    def test_training_loop(self, batch_size=16):

        torch.autograd.set_detect_anomaly(True)

        gen2_output_dim = 20 # output dimension of the second generator for simulator parameters
        max_beat_length = 50 # maximum length of the beat vector
        noise_dim = 50 # dimension of the noise vector
        adj_size = (64,64) # dimensions of the adjacency matrix
        
        start = 100 # start of the sequence
        sequence_length = 50 # length of the sequence generated by simulator   NOTE: TO CHANGE THIS, YOU NEED TO RE-PICKLE THE DATASET WITH THE NEW SEQUENCE LENGTH
        
        roll_size = (2,128,sequence_length) # dimensions of the piano roll matrix First channel is note velocity, second channel is time step


        # get the correct device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Starting dataset creation...")
        start = time.time()
        dataset = MaestroDatasetPickle(f'preprocessed_data_{sequence_length}.pkl', beats_length=max_beat_length, sequence_length=sequence_length, device=device)
        train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
        print("Dataset creation took", time.time() - start, "seconds")  

        print("Starting setup of model...")
        start = time.time()

        # Initialize the model
        mmgan = MultiModalGAN(z_dim=noise_dim, adj_size=adj_size, roll_size=roll_size, input_dim=max_beat_length, output_dim=gen2_output_dim, instrument=0, start=start, end=start+sequence_length, device=device)

        # Check if a saved model exists and load it
        model_path = 'models\MAE_loss\mmgan_64_64_epoch_35.pth'   # CHANGE THIS TO THE PATH OF YOUR SAVED MODEL WHEN YOU WANT TO CONTINUE TRAINING
        if os.path.isfile(model_path):
            mmgan.load_state_dict(torch.load(model_path))
            print("Loaded model from", model_path)
        else:
            print("No saved model found, starting training from scratch")

        criterion = nn.BCEWithLogitsLoss()
        #criterion = nn.MSELoss()
        #criterion = nn.L1Loss()

        # Create separate optimizers for the generator and the discriminator
        gen_opt = torch.optim.Adam(list(mmgan.generator1.parameters()) + list(mmgan.generator2.parameters()), lr=0.01)
        disc_opt = torch.optim.Adam(mmgan.discriminator.parameters(), lr=0.01)

        # Create learning rate schedulers for the generator and the discriminator
        gen_scheduler = StepLR(gen_opt, step_size=30, gamma=0.1)
        disc_scheduler = StepLR(disc_opt, step_size=30, gamma=0.1)

        print("Model setup took", time.time() - start, "seconds")

        num_epochs = 100
        print_interval = 10
        save_interval = 5  # Save the model every 5 epochs

        count = 0

        total_failures = 0
        total_seen = 0

        gen_loss_history = []
        disc_loss_history = []

        for epoch in range(num_epochs):
            mmgan.train()
            disc_losses = []
            gen_losses = []

            for i, (piano_roll, durations, beats) in enumerate(train_loader):
                count += 1

                noise1 = torch.randn(batch_size, noise_dim, device=device)
                noise2 = torch.randn(batch_size, noise_dim, device=device)
                real = torch.ones(batch_size).to(device)
                fake_label = torch.zeros(batch_size).to(device)

                # Load pickled tensors
                real_data = torch.stack([piano_roll.to(device), durations.to(device)]).permute(1, 0, 2, 3)

                # Train Discriminator
                disc_opt.zero_grad()
                fake_output, failed_sim_count = mmgan(noise1, noise2, beats, count)
                disc_fake_loss = criterion(fake_output.squeeze(), fake_label)
                disc_real_loss = criterion(mmgan.discriminator(real_data).squeeze(), real)
                disc_loss = disc_fake_loss + disc_real_loss
                disc_loss.backward()
                disc_opt.step()

                # Train both generators
                gen_opt.zero_grad()
                fake_output, failed_sim_count = mmgan(noise1, noise2, beats, count)
                gen_loss = criterion(fake_output.squeeze(), real)
                gen_loss.backward()
                gen_opt.step()

                total_failures += failed_sim_count
                total_seen += batch_size

                disc_losses.append(disc_loss.item())
                gen_losses.append(gen_loss.item())

                if i % 5 == 0:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Batch {i}/{len(train_loader)}, Avg Disc Loss: {sum(disc_losses) / len(disc_losses)}, Avg Gen Loss: {sum(gen_losses) / len(gen_losses)}')
                    print("Total failures:", total_failures, "Total seen:", total_seen)
    

            disc_scheduler.step()
            gen_scheduler.step()

            with open(f'losses/disc_losses_epoch_{epoch + 1}.pkl', 'wb') as f:
                pickle.dump(disc_losses, f)
            with open(f'losses/gen_losses_epoch_{epoch + 1}.pkl', 'wb') as f:
                pickle.dump(gen_losses, f)

            if (epoch + 1) % print_interval == 0:

                print(f'Epoch {epoch + 1}/{num_epochs}, Avg Disc Loss: {sum(disc_losses) / len(disc_losses)}, Avg Gen Loss: {sum(gen_losses) / len(gen_losses)}')
            
            print("Total failures:", total_failures, "Total seen:", total_seen)


            # timeout for 10 seconds
            time.sleep(10)

            # Save the model every save_interval epochs
            if (epoch + 1) % save_interval == 0:
                torch.save(mmgan.state_dict(), f'models//mmgan_{adj_size[0]}_{adj_size[1]}_epoch_{epoch + 1}.pth')

        return disc_losses, gen_losses
    
if __name__ == '__main__':
    unittest.main()