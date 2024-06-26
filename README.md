# LoRA-from-scratch
This repository contains a PyTorch implementation of Low-Rank Adaptation (LoRA), applied to the task of classifying MNIST digits. The implementation demonstrates how LoRA can be integrated into a neural network and fine-tuned on specific tasks, allowing for efficient training and memory optimizations.

## How LoRA Works

LoRA introduces two small matrices, A and B, which together approximate the weight update matrix ΔW . The inner dimension r of these matrices is a hyperparameter that controls the rank and complexity of the approximation. This technique modifies the standard training process by updating only these smaller matrices, rather than the entire weight matrix, which can significantly reduce memory usage and computational costs.

![LoRA Weight Update Diagram](media/LoRA_diagram.jpg)

## Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/lightmatmul/LoRA-from-scratch.git
cd LoRA-from-scratch
```

Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To run the training and testing scripts, use the following command:

```bash
python main.py
```

This command will:

- Train a neural network on the MNIST dataset (simulating LLM pretraining).
- Fine-tune the network on a poorly performing digit with and without LoRA.
- Test the two fine-tunes to compare performances and demonstrate LoRA's efficiency.
