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



def display_images(image_tensor, num_images=25, size=(1, 28, 28)):
    flatten_image = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(flatten_image[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_noise(n_samples, noise_dim, device='cpu'):
    return torch.randn(n_samples, noise_dim, device=device)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        torch.nn.init.constant_(m.bias, val=0.5)
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        torch.nn.init.constant_(m.bias, val=0.5)


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
            nn.ReLU(inplace=True),
        )

    def forward(self, noise):
        return self.gen(noise).view(len(noise), -1, self.adj_size[0], self.adj_size[1]).to(self.device)  # change the order of dimensions


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
            nn.ReLU(inplace=True),
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
            self.make_disc_block(im_chan * roll_size[0] * roll_size[1], hidden_dim),
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
            

class MultiModalGAN(nn.Module):
    def __init__(self, z_dim=100, hidden_dim=64, adj_size=(28, 28),roll_size=(128,100), input_dim=50, output_dim=16, instrument=None, start=0, end=100, device='cpu'):
        super(MultiModalGAN, self).__init__()
        self.z_dim = z_dim
        self.generator1 = Generator(z_dim, hidden_dim=hidden_dim, adj_size=adj_size, device=device).to(device)
        self.generator2 = BeatGenerator(z_dim, hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim, device=device).to(device)
        self.discriminator = Discriminator(roll_size=roll_size, device=device).to(device)
        self.instrument = instrument
        self.start = start
        self.end = end
        self.adj_size = adj_size 
        self.device = device

    def forward(self, noise1, noise2, input_tensor, count):
        gen_output1 = self.generator1(noise1)
        gen_output2 = self.generator2(noise2, input_tensor)

        start = time.time()
        sim_midi = matrix_to_midi(gen_output1, gen_output2, adj_size=self.adj_size, instrument=self.instrument, start=self.start, end=self.end, count=count)
        print("matrix_to_midi took", time.time() - start, "seconds")

        sim_midi = torch.stack([torch.Tensor(x) for x in sim_midi]).to(self.device)


        print("sim_midi.shape", sim_midi.shape)

        sim_midi = sim_midi.view(len(sim_midi), -1)

        disc_output = self.discriminator(sim_midi)

        return disc_output

class TestMultiModalGAN(unittest.TestCase):
    def test_training_loop(self, batch_size=16):

        max_beat_length = 50 # maximum length of the beat vector
        noise_dim = 50 # dimension of the noise vector
        adj_size = (28,28) # dimensions of the adjacency matrix
        roll_size = (128,100) # dimensions of the piano roll matrix

        start = 0 # start of the sequence
        sequence_length = 100 # length of the sequence 

        gen2_output_dim = 20 # output dimension of the second generator
        

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
        print("Model setup took", time.time() - start, "seconds")

        num_epochs = 3
        print_interval = 10

        count = 0

        for epoch in range(num_epochs):
            mmgan.train()
            train_losses = []
            for i, (piano_roll, durations, beats) in enumerate(train_loader):
                print("Batch", count)
                count += 1
                noise1 = torch.randn(batch_size, noise_dim, device=device)
                noise2 = torch.randn(batch_size, noise_dim, device=device)
                real = torch.ones(batch_size, 1, device=device).view(-1)
                fake = mmgan(noise1, noise2, beats, count).squeeze(1)

                start = time.time()
                print("Updating weights...")
                gen_loss = criterion(fake, real)
                gen_loss.backward()
                gen_opt.step()
                gen_opt.zero_grad()
                scheduler.step()
                print("Updating weights took", time.time() - start, "seconds")

                train_losses.append(gen_loss.item())
                if (i + 1) % print_interval == 0:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Step {i + 1}/{len(train_loader)}, Train Loss: {gen_loss.item()}')

            print(f'Epoch {epoch + 1}/{num_epochs}, Avg Train Loss: {sum(train_losses) / len(train_losses)}')

        return train_losses
    
if __name__ == '__main__':
    unittest.main()