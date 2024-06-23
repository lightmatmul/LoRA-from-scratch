import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from models import NeuralNet
from utils import train, test, enable_disable_lora, count_trainable_params


def main():
    # Data transformations and loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(full_train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    model_with_lora = NeuralNet().to(device)
    model_without_lora = NeuralNet().to(device)

    # Clone the weights from model_with_lora to model_without_lora
    model_without_lora.load_state_dict(model_with_lora.state_dict())

    # Set up the optimizer and loss function
    optimizer_with_lora = optim.Adam(model_with_lora.parameters(), lr=0.001)
    optimizer_without_lora = optim.Adam(model_without_lora.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initial full training
    print("Starting training on full dataset...")
    train(train_loader, model_with_lora, device, optimizer_with_lora, criterion, epochs=1)
    
    # Report number of trainable parameters before fine-tuning
    print(f"Trainable parameters with LoRA: {count_trainable_params(model_with_lora)}")
    
    # Clone original weights for comparison after fine-tuning
    original_weights = {name: param.clone().detach() for name, param in model_with_lora.named_parameters()}

    # Fine-tuning on poorly performing digit
    digit_to_finetune = 9
    indices = [i for i, y in enumerate(full_train_dataset.targets) if y == digit_to_finetune]
    finetune_dataset = Subset(full_train_dataset, indices)
    finetune_loader = DataLoader(finetune_dataset, batch_size=10, shuffle=True)

    print(f"Fine-tuning on digit {digit_to_finetune} with LoRA enabled...")
    enable_disable_lora(model_with_lora, enabled=True)
    train(finetune_loader, model_with_lora, device, optimizer_with_lora, criterion, epochs=1)

    print(f"Fine-tuning on digit {digit_to_finetune} with LoRA disabled...")
    enable_disable_lora(model_without_lora, enabled=False)  # Ensure LoRA is disabled
    train(finetune_loader, model_without_lora, device, optimizer_without_lora, criterion, epochs=1)

    # Report number of trainable parameters after fine-tuning with LoRA disabled
    print(f"Trainable parameters without LoRA: {count_trainable_params(model_without_lora)}")

    # Testing with LoRA enabled
    print("Testing with LoRA enabled...")
    test(test_loader, model_with_lora, device)

    # Testing with LoRA disabled
    print("Testing with LoRA disabled...")
    test(test_loader, model_without_lora, device)

if __name__ == "__main__":
    main()