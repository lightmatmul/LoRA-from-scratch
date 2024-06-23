import torch
import torch.nn as nn
from tqdm import tqdm

class LoRAParametrization(nn.Module):
    """
    Implements LoRA (Low-Rank Adaptation) to modify layer weights during training.
    """
    def __init__(self, features_in, features_out, rank=1, alpha=1, device='cpu'):
        super().__init__()
        # Initialize low-rank matrices A and B
        # A random Gaussian initialization for A and zero for B, so ∆W = BA is zero at the beginning of training
        self.lora_A = nn.Parameter(torch.randn(rank, features_out, device=device))
        self.lora_B = nn.Parameter(torch.zeros(features_in, rank, device=device))

        #   We then scale ∆Wx by α/r , where α is a constant in r. 
        #   When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately. 
        #   As a result, α is set to the first r we try and do not tune it. 
        #   This scaling helps to reduce the need to retune hyperparameters when we vary r.
        self.scale = alpha / rank  # Scale delta weights
        self.enabled = True  # Controls if LoRA is applied or original weights are used

    def forward(self, original_weights):
        # Modify original weights based on LoRA parameters if enabled
        if self.enabled:
            delta_weights = torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape)
            return original_weights + delta_weights * self.scale
        return original_weights

    @staticmethod
    def add_to_layer(layer, device, rank=1, alpha=1):
        """
        Adds LoRA parameterization to a neural network layer.
        """
        # Capture input/output features of the layer to size LoRA matrices appropriately
        features_in, features_out = layer.weight.shape
        parametrization = LoRAParametrization(features_in, features_out, rank, alpha, device)
        nn.utils.parametrize.register_parametrization(layer, 'weight', parametrization)

def train(train_loader, net, device, optimizer, criterion, epochs=1):
    """
    Trains a neural network for a specified number of epochs.
    """
    net.train()  # Set model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)  # Move data to the appropriate device
            optimizer.zero_grad()  # Clear gradients before each backpropagation
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()  # Compute gradient
            optimizer.step()  # Update weights
            total_loss += loss.item()  # Aggregate loss for reporting
        print(f"Epoch {epoch+1}: Average Loss: {total_loss / len(train_loader)}")

def test(test_loader, net, device):
    """
    Evaluates the network on the test set and prints accuracy and wrong counts per digit.
    """
    net.eval()  # Set model to evaluation mode
    total = 0
    correct = 0
    wrong_counts = [0] * 10
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for x, y in tqdm(test_loader, desc='Testing'):
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            for label, prediction in zip(y, predicted):
                if label != prediction:
                    wrong_counts[label.item()] += 1  # Count misclassifications for each digit

    accuracy = correct / total * 100
    print(f'Accuracy: {accuracy:.2f}%')
    for i, count in enumerate(wrong_counts):
        print(f'Digit {i}: {count} errors')

def enable_disable_lora(net, enabled=True):
    """
    Enables or disables the LoRA adaptations across the network.
    """
    # Toggle the 'enabled' status of LoRA layers throughout the network
    for name, module in net.named_modules():
        if hasattr(module, 'enabled'):
            module.enabled = enabled

def count_trainable_params(model):
    """Helper function to count the model's trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)