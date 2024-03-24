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
from matrix_sim_process import matrix_to_wav

from datasets import InputSong, MaestroDataset, my_collate


def display_images(image_tensor, num_images=25, size=(1, 28, 28)):
    flatten_image = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(flatten_image[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_noise(n_samples, noise_dim, device='cpu'):
    '''
    Generate noise vectors from the random normal distribution with dimensions (n_samples, noise_dim),
    where
        n_samples: the number of samples to generate based on  batch_size
        noise_dim: the dimension of the noise vector
        device: device type can be cuda or cpu
    '''

    return torch.randn(n_samples, noise_dim, 1, 1, device=device)


def weights_init(m):
    """
    Initialize weights to a normal distribution with a mean of 0 and standard deviation of 0.02
    :param m: model
    :return: model with initialized weights
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, val=0)


class Generator(nn.Module):
    """"
    The Generator takes the noise vector as an input to generate the images that will resemble the training dataset(1,28,28) accomplished through a series of strided two-dimensional ConvTranspose2d layers. The ConvTranspose2d layers are paired with BatchNorm2d layers as they help with the flow of gradients during training, which is followed by a ReLU activation function.
    The output of the Generator is a tanh activation function to have the pixel values in the range of [-1,1]
    """

    def __init__(self, no_of_channels=1, noise_dim=100, gen_dim=32):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=noise_dim, out_channels=gen_dim * 4, kernel_size=4, stride=1,
                                        padding=0,
                                        bias=False)
        self.conv2 = nn.ConvTranspose2d(in_channels=gen_dim * 4, out_channels=gen_dim * 2, kernel_size=4, stride=2,
                                        padding=1,
                                        bias=False)
        self.conv3 = nn.ConvTranspose2d(in_channels=gen_dim * 2, out_channels=gen_dim, kernel_size=4, stride=2,
                                        padding=1,
                                        bias=False)
        self.conv4 = nn.ConvTranspose2d(in_channels=gen_dim, out_channels=no_of_channels, kernel_size=5, stride=1,
                                        padding=0,
                                        bias=False)
        self.batch_norm1 = nn.BatchNorm2d(gen_dim * 4)
        self.batch_norm2 = nn.BatchNorm2d(gen_dim * 2)
        self.batch_norm3 = nn.BatchNorm2d(gen_dim)

        # Initialize weights to random values
        self._initialize_weights()

    def _initialize_weights(self):  # suggested by copilot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight, 1.0, 0.02)
                init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        forward pass of the generator:
        Input is a noise tensor and output is generated images resembling Training data.
        '''
        # output = self.network(input)
        # x = torch.relu(self.batch_norm1(self.conv1(input)))
        x = self.conv1(input)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = torch.relu(self.batch_norm3(self.conv3(x)))
        x = self.conv4(x)
        output = torch.sigmoid(x)  # changed from tanh to sigmoid for better

        return output


class Discriminator(nn.Module):
    """
    Takes as input a tensor of dimensions (batch_size, 128, 216) and returns probabilities for each element in the
    batch, resulting in a tensor of size (batch_size, 1)
    """

    def __init__(self, no_of_channels=1, disc_dim=32):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 32 * 54, 128)  # Calculate the size based on input dimensions
        self.fc2 = nn.Linear(128, 1)  # Output 1 for binary classification

    def forward(self, input):
        '''
        forward pass of the discriminator
        Input is an image tensor,
        returns a 1-dimension tensor representing image as
        fake/real.
        '''
        x = torch.unsqueeze(input, 1)  # (batch_size, 1, 128, 216)
        x = self.pool(torch.relu(self.conv1(x)))  # (batch_size, 16, 64, 108)
        x = self.pool(torch.relu(self.conv2(x)))  # (batch_size, 32, 32, 54)
        x = x.view(-1, 32 * 32 * 54)  # Flatten the tensor for fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary classification
        return x  # (batch_size, 1)


class SimNN(nn.Module):
    def __init__(self, n):
        super(SimNN, self).__init__()
        self.n = n
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # this will be adjusted in forward
        self.fc2 = nn.Linear(512, self.n * self.n + 4 * self.n)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        # adjust the size of the input to the first fully connected layer
        self.fc1 = nn.Linear(x.size(1), 512).to(x.device)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        matrix = output[:, :self.n * self.n].view(-1, self.n, self.n)
        array1 = output[:, self.n * self.n:self.n * self.n + self.n]
        array2 = output[:, self.n * self.n + self.n:self.n * self.n + 2 * self.n]
        array3 = output[:, self.n * self.n + 2 * self.n:self.n * self.n + 3 * self.n]
        array4 = output[:, self.n * self.n + 3 * self.n:]
        return matrix, array1, array2, array3, array4

    def create_model(n):
        model = SimNN(n)
        return model

    def pretrain_model(model, pretrain_dataloader, error_system, num_epochs=5):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        for epoch in range(num_epochs):
            for spectrograms, targets in pretrain_dataloader:
                optimizer.zero_grad()
                matrix, array1, array2, array3, array4 = model(spectrograms)
                outputs = [matrix, array1, array2, array3, array4]
                loss = 0
                for output, target in zip(outputs, targets):
                    # simulate the system with the output and get the error
                    error = error_system.simulate(output)
                    # use the error as the target for the loss function
                    loss += criterion(output, error)
                loss.backward()
                optimizer.step()

    def error_system(output):
        # This function should simulate the system with the given output and return the error.
        # This is a placehold
        numpy_melspectro_output = matrix_to_wav([output])
        error = None
        return error


def test_SimNN():
    n = 10  # adjust as needed
    model = SimNN(n)
    batch_size = 16
    for _ in range(50):  # test with 5 different sizes
        size = torch.randint(128, 32769, (1,)).item()  # random size between 128 and 512
        input = torch.randn(batch_size, 1, size, size)
        matrix, array1, array2, array3, array4 = model(input)
        assert matrix.size() == (batch_size, n, n)
        assert array1.size() == (batch_size, n)
        assert array2.size() == (batch_size, n)
        assert array3.size() == (batch_size, n)
        assert array4.size() == (batch_size, n)
    print("All tests passed.")


if __name__ == '__main__':
    # test_SimNN()
    # TEST
    # Transform the images to tensors and normalize them
    my_transform = transforms.Compose([transforms.Resize(28),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=(0.5,), std=(0.5,))])
    # Load MNIST dataset as tensors
    batch_size = 32  # number of training samples in one iteration TODO: what is a good batch size

    # dataloader = DataLoader(
    #     datasets.MNIST('.', download=True, transform=my_transform),
    #     batch_size=batch_size,
    #     shuffle=True)

    # input_song = InputSong(audio_file="./data/classical.00000.wav")

    MODEL_PATH = "models/"
    input_song = MaestroDataset()
    dataloader = DataLoader(input_song, batch_size=2, shuffle=True, collate_fn=my_collate)

    # START
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # put both models on the device
    gen = Generator().to(device)
    disc = Discriminator().to(device)

    # initialize the weights for Conv2d, ConvTranspose2d and BatchNorm2d
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    # define loss and optimizer as binary cross-entropy loss and adam optimizer
    lr = 0.00002
    criterion = nn.BCEWithLogitsLoss()
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    # train params
    n_epochs = 5
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    display_step = 20
    save_step = 20
    z_dim = 100

    # loss histories
    gen_losses = list()
    disc_losses = list()

    # start training
    for epoch in range(n_epochs):
        for real in tqdm(dataloader):  # Dataloader returns the batches
            cur_batch_size = len(real)
            real = real.to(device)

            # DISCRIMINATOR: get loss on real images
            # train discriminator on batch of real images from the training dataset
            disc_opt.zero_grad()
            disc_real_pred = disc(real).reshape(-1)
            real_label = (torch.ones(cur_batch_size) * 0.9).to(device)
            # real_label = torch.ones(disc_real_pred.shape, device=device)

            # Get the discriminator's prediction on the real image and
            # calculate the discriminator's loss on real images
            disc_real_loss = criterion(disc_real_pred, real_label)

            # DISCRIMINATOR: get loss on fake images
            # generate the random noise
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)

            # generate the fake images by passing the random noise to the generator
            fake = gen(fake_noise)  # matrix of size (batch_size, 1 , 20, 20)

            # convert generated tensors to list of numpy matrices
            fake = fake.squeeze().detach().cpu().numpy()

            fake = matrix_to_wav(fake, start=0, end=216,
                                 device=device)  # build simulator from matrix and get spectrograms from created audio
            # transform list of length batch_size with (128, 174) shaped matrices to tensor (batch_size, 128, 174)

            print(fake.shape)

            # Get the discriminator's prediction on the fake images generated by generator
            disc_fake_pred = disc(fake.detach()).reshape(-1)

            fake_label = (torch.ones(cur_batch_size) * 0.1).to(device)

            # calculate the discriminator's loss on fake images
            disc_fake_loss = criterion(disc_fake_pred, fake_label)

            # DISCRIMINATOR: final loss computation
            disc_loss = (disc_fake_loss + disc_real_loss)
            disc_loss.backward()  # update gradients
            disc_opt.step()  # update optimizer

            # Keep track of the discriminator loss
            disc_losses.append(disc_loss.item())

            # GENERATOR
            gen_opt.zero_grad()

            # Get the discriminator's prediction on the fake images
            disc_fake_pred = disc(fake).squeeze()
            real_label = (torch.ones(cur_batch_size)).to(device)

            # GENERATOR: loss calculation and update
            gen_loss = criterion(disc_fake_pred, real_label)
            gen_loss.backward()  # update gradients
            gen_opt.step()  # update optimizer

            # Keep track of the generator loss
            gen_losses.append(gen_loss.item())

            ## Visualization code ##
            if cur_step % display_step == 0 and cur_step > 0:
                mean_generator_loss = np.mean(gen_losses)
                mean_discriminator_loss = np.mean(disc_losses)
                print(
                    f"Epoch:{epoch} Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: "
                    f"{mean_discriminator_loss}")
            # save model weights
            if cur_step % save_step == 0 and cur_step > 0:
                torch.save(gen.state_dict(), f"{MODEL_PATH}/gen_{cur_step}_{datetime.datetime.now()}")  # save generator
            cur_step += 1
