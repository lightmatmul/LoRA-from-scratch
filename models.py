import torch
import torch.nn as nn
from utils import LoRAParametrization

class NeuralNet(nn.Module):
    """
    An NN designed for MNIST classification to demonstrate the effect of LoRA.
    """
    def __init__(self, hidden_size_1=1000, hidden_size_2=2000):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(28*28, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()

        # Initialize LoRA modifications
        self.add_lora_parametrizations()

    def forward(self, img):
        """
        Forward pass through the network.
        """
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def add_lora_parametrizations(self):
        """
        Adds LoRA parameterizations to each linear layer.
        """
        # Retrieve the device to place LoRA parameters on the same device as the model
        device = next(self.parameters()).device
        
        # Add LoRA to each linear layer with default settings (can be adjusted)
        LoRAParametrization.add_to_layer(self.linear1, device)
        LoRAParametrization.add_to_layer(self.linear2, device)
        LoRAParametrization.add_to_layer(self.linear3, device)

