# MNIST Handwritten Digit Classification Neural Network

A from-scratch implementation of a neural network for handwritten digit recognition, following the 3Blue1Brown Neural Networks series. Built using PyTorch with a focus on understanding the fundamental concepts of deep learning.

## Project Overview

This project implements a fully connected neural network that learns to classify handwritten digits (0-9) from the MNIST dataset. The goal was to understand neural network fundamentals rather than achieve state-of-the-art performance, following Grant Sanderson's excellent educational approach in his 3Blue1Brown series.

## Results

- **Test Accuracy**: 85.73%
- **Training Loss**: Decreased from ~2.3 to ~0.6 over 20 epochs
- **Architecture**: 784 → 16 → 16 → 10 neurons
- **Training Behavior**: Smooth loss curve indicating stable learning

## Architecture

```
Input Layer:    784 neurons (28×28 flattened pixels)
Hidden Layer 1: 16 neurons + Sigmoid activation
Hidden Layer 2: 16 neurons + Sigmoid activation  
Output Layer:   10 neurons (digit probabilities)
```

### Why This Architecture?

- **Input size (784)**: Each MNIST image is 28×28 pixels, flattened to a vector
- **Hidden layers (16×16)**: Following 3Blue1Brown's example, designed to learn hierarchical features
- **Sigmoid activation**: Prevents linear collapse, enables non-linear pattern recognition
- **Output size (10)**: One neuron per digit class (0-9)

## Key Concepts Implemented

### Neural Network Fundamentals

- **Forward Propagation**: Data flows through layers with linear transformations and activations
- **Backpropagation**: Automatic gradient calculation using PyTorch's `loss.backward()`
- **Gradient Descent**: SGD optimizer updates weights to minimize cross-entropy loss

### Critical Insights Learned

- **Activation Functions Are Essential**: Without sigmoid between layers, the entire network collapses to a single linear equation
- **Batch Processing**: Training on mini-batches (32 images) provides efficient learning with ~1,875 weight updates per epoch
- **Loss vs Accuracy**: Cross-entropy loss provides a continuous measure of prediction quality

## Implementation Details

### Technologies Used

- **PyTorch**: Neural network framework with automatic differentiation
- **Google Colab**: Development environment with free GPU access
- **torchvision**: MNIST dataset loading and preprocessing

### Key Components

**Data Pipeline:**

```python
transform = transforms.ToTensor()  # Normalize [0,255] → [0,1]
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

**Network Architecture:**

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 16)
        self.layer2 = nn.Linear(16, 16) 
        self.layer3 = nn.Linear(16, 10)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.layer3(x)  # Raw scores for cross-entropy
        return x
```

**Training Loop:**

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    for batch_images, batch_labels in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_images)
        loss = F.cross_entropy(predictions, batch_labels)
        loss.backward()
        optimizer.step()
```

## Training Process

### Loss Function

- **Cross-Entropy Loss**: Measures prediction quality for classification
- **Why Cross-Entropy**: Penalizes confident wrong answers more than uncertain ones
- **PyTorch Integration**: `F.cross_entropy()` handles softmax conversion internally

### Optimization

- **SGD (Stochastic Gradient Descent)**: Basic but reliable optimization algorithm
- **Learning Rate**: 0.01 provided stable convergence
- **Batch Size**: 32 images per batch balanced memory usage and gradient quality

### Results Analysis

- **Initial Performance**: ~2.3 loss (random guessing baseline)
- **Final Performance**: ~0.6 loss, 85.73% test accuracy
- **Learning Curve**: Smooth decrease over 20 epochs indicates healthy training

## What the Network Actually Learned

### The Reality Check

While we hoped the layers would learn:

- Layer 1: Edge detection
- Layer 2: Shape/loop detection

**The truth is**: We don't know exactly what internal representations the network discovered! The 85.73% accuracy proves it learned something useful, but neural networks are "black boxes" that find their own solutions through gradient descent.

### Example Predictions

The network makes sensible mistakes - for instance, predicting a poorly written "5" as a "6" when the handwriting is genuinely ambiguous.

## Future Improvements

### Immediate Experiments

- **Architecture variations**: Try [784, 32, 32, 10] or [784, 16, 16, 16, 10]
- **Activation functions**: Replace sigmoid with ReLU
- **Optimizers**: Experiment with Adam instead of SGD
- **Learning rate scheduling**: Decay learning rate over time

### Next-Level Approaches

- **Convolutional Neural Networks (CNNs)**: Better suited for images, typically achieve 98-99% on MNIST
- **Data augmentation**: Rotation, scaling, noise to improve generalization
- **Regularization**: Dropout and batch normalization to prevent overfitting

## Learning Resources

This project was inspired by and follows:

- [3Blue1Brown Neural Networks Series](https://www.3blue1brown.com/topics/neural-networks)
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)

## Educational Value

### Concepts Mastered

- Neural network mathematical foundations
- PyTorch framework and tensor operations
- Gradient descent and backpropagation intuition
- Training loop design and debugging
- Model evaluation and performance analysis
- Understanding of activation functions and network depth

### Skills Developed

- Implementing neural networks from mathematical principles
- Data pipeline design for machine learning
- Training loop optimization and monitoring
- Model evaluation and performance interpretation
- Critical thinking about black box models

## Running the Code

### Requirements

```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.21.0
```

### Quick Start

1. Open in Google Colab for free GPU access
2. Install dependencies: `pip install torch torchvision matplotlib`
3. Run the training loop and monitor loss progression
4. Evaluate on test set to measure generalization

### Expected Output

- Training loss should decrease from ~2.3 to ~0.6
- Test accuracy should reach 80-90% range
- Training time: ~10-15 minutes on GPU

## Performance Context

|Approach|Typical MNIST Accuracy|
|---|---|
|Random Guessing|~10%|
|Linear Model|~92%|
|**Our Neural Network**|**85.73%**|
|Simple CNN|~98%|
|Advanced CNN|~99.5%+|

## Key Takeaways

1. **Activation functions prevent linear collapse** - crucial insight for deep learning
2. **Batch processing enables efficient training** - fundamental for practical ML
3. **Neural networks are powerful function approximators** - but remain black boxes
4. **Good engineering practices matter** - proper train/test splits, monitoring, evaluation
5. **Understanding foundations enables innovation** - theory enables practical problem-solving

---

_This project represents a complete journey from neural network theory to working implementation, emphasizing understanding over performance optimization._