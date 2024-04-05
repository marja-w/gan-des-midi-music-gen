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

from util import get_melspectrogram_db_from_file, get_melspectrogram_db_tensor
import scipy.io.wavfile as wav

from datasets import MaestroPianoRollDataset  # TODO: does not find MaestroPianoRollDataset

#from network_tests import display_images, get_noise, weights_init
#from network_tests import Generator, Discriminator

def display_images(image_tensor, num_images=25, size=(1, 28, 28)):
    flatten_image = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(flatten_image[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_noise(n_samples, noise_dim, device='cpu'):
    return torch.randn(n_samples, noise_dim, device=device)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, val=0)


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64, image_size=None):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.image_size = image_size  # store image_size
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan * image_size[0] * image_size[1]),
        )

    def make_gen_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, noise):
        return self.gen(noise).view(len(noise), -1, self.image_size[0], self.image_size[1])  # change the order of dimensions


class BeatGenerator(nn.Module):
    def __init__(self, z_dim=10, hidden_dim=64,  input_dim=None, output_dim=None):
        super(BeatGenerator, self).__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim  # store output_dim
        self.input_tensor_dim = input_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim + input_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, output_dim),
        )

    def make_gen_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, noise, input_tensor=None):
        # generate random tensor of length input_densor_dim if input_tensor is None
        if input_tensor == None:
            input_tensor = torch.randn(len(noise), self.input_tensor_dim) 

        combined_input = torch.cat((noise, input_tensor), dim=1)
        return self.gen(combined_input)


class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=16, image_size=None):
        super(Discriminator, self).__init__()
        self.image_size = image_size  # store image_size
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan * image_size[0] * image_size[1], hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1),
        )

    def make_disc_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image):
        return self.disc(image.view(len(image), -1)).view(-1)  # remove the extra dimension
    

class MultiModalGAN(nn.Module):
    def __init__(self, z_dim=10, hidden_dim=64, image_size=None, input_dim=None, output_dim=None):
        super(MultiModalGAN, self).__init__()
        self.generator1 = Generator(z_dim, hidden_dim=hidden_dim, image_size=image_size)
        self.generator2 = BeatGenerator(z_dim, hidden_dim=hidden_dim, input_dim=input_dim, output_dim=input_dim)
        self.discriminator = Discriminator(image_size=image_size)  # pass image_size to Discriminator


        self.image_size = image_size  # store image_size ( CAN BE REMOVED WHEN SIM OUTPUT USED)

    def forward(self, noise1, noise2, input_tensor):
        gen_output1 = self.generator1(noise1)
        gen_output2 = self.generator2(noise2, input_tensor)
        # Directly pass the outputs of the generators to the discriminator

        # create random noise of image_size
        noise = torch.randn(len(noise1), *self.image_size)  # USE SIM HERE

        disc_output = self.discriminator(noise)
        return disc_output
    

if __name__ == '__main__':
    # Load Piano Roll dataset as tensors
    midi_files = ['adj_sim_outputs\midi\output.mid', 'adj_sim_outputs\midi\output.mid', 'adj_sim_outputs\midi\output.mid']  # replace with your actual file paths
    dataset = MaestroPianoRollDataset(midi_files)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # START
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    noise_dim = 100
    image_size = (20, 20)  # replace with the desired image size
    beat_dim = 20
    output_dim = 20  # use the same dimensions as the input for tests

    # put both models on the device
    mmgan = MultiModalGAN(noise_dim, image_size=image_size, input_dim=beat_dim, output_dim=output_dim.to(device))
    # initialize the weights for Conv2d, ConvTranspose2d and BatchNorm2d
    mmgan = mmgan.apply(weights_init)

    MODEL_PATH = "models/"

    # define loss and optimizer as binary cross-entropy loss and adam optimizer
    lr = 0.00002
    criterion = nn.BCEWithLogitsLoss()
    gen_opt = torch.optim.Adam(mmgan.parameters(), lr=lr, betas=(0.5, 0.999))

    # train params
    n_epochs = 5
    cur_step = 0
    mean_generator_loss = 0
    display_step = 20
    save_step = 20
    z_dim = 100

    # loss histories
    gen_losses = list()

    # start training
    for epoch in range(n_epochs):
        for real, durations, beats in tqdm(dataloader):  # Dataloader returns the batches
            cur_batch_size = len(real)
            real = real.to(device)
            beats = beats.to(device)

            # GENERATOR: get loss on fake images
            # generate the random noise
            fake_noise1 = get_noise(cur_batch_size, z_dim, device=device)
            fake_noise2 = get_noise(cur_batch_size, z_dim, device=device)

            # generate the fake images by passing the random noise to the generator
            fake = mmgan(fake_noise1, fake_noise2, beats)  # matrix of size (batch_size, 1 , 20, 20)

            # GENERATOR: loss calculation and update
            gen_loss = criterion(fake, real)
            gen_loss.backward()  # update gradients
            gen_opt.step()  # update optimizer

            # Keep track of the generator loss
            gen_losses.append(gen_loss.item())

            ## Visualization code ##
            if cur_step % display_step == 0 and cur_step > 0:
                mean_generator_loss = np.mean(gen_losses)
                print(f"Epoch:{epoch} Step {cur_step}: Generator loss: {mean_generator_loss}")
            # save model weights
            if cur_step % save_step == 0 and cur_step > 0:
                torch.save(mmgan.state_dict(), f"{MODEL_PATH}/mmgan_{cur_step}_{datetime.datetime.now()}")  # save generator
            cur_step += 1