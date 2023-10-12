import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda

# Check if GPU is available, and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Create some random data and move it to the GPU if available
input_data = torch.randn(64, 10).to(device)
target_data = torch.randn(64, 1).to(device)

# Create an instance of the neural network and move it to the GPU
net = SimpleNet().to(device)

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = net(input_data)
    loss = criterion(output, target_data)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

print("Training complete!")
