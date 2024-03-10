
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class SimNN(nn.Module):
    def __init__(self, n):
        super(SimNN, self).__init__()
        self.n = n
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # this will be adjusted in forward
        self.fc2 = nn.Linear(512, self.n*self.n + 4*self.n)

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
        matrix = output[:, :self.n*self.n].view(-1, self.n, self.n)
        array1 = output[:, self.n*self.n:self.n*self.n+self.n]
        array2 = output[:, self.n*self.n+self.n:self.n*self.n+2*self.n]
        array3 = output[:, self.n*self.n+2*self.n:self.n*self.n+3*self.n]
        array4 = output[:, self.n*self.n+3*self.n:]
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

test_SimNN()