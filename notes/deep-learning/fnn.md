## II. Feedforward Neural Networks (FNN) / Multi-Layer Perceptrons (MLP)

Feedforward Neural Networks, also known as Multi-Layer Perceptrons (MLPs), are the simplest type of artificial neural network. They form the foundation for many more complex deep learning architectures.

### 1. Origin and Basic Structure

*   **Origin:**
    *   The concept of an FNN builds upon the **Perceptron**, developed by Frank Rosenblatt in the 1950s, which was a single-layer neural network capable of learning binary classifiers for linearly separable data.
    *   The limitations of single-layer perceptrons (e.g., inability to solve the XOR problem) led to the development of multi-layer perceptrons, which introduce "hidden" layers between the input and output layers.
    *   The backpropagation algorithm, popularized in the 1980s, provided an efficient way to train these multi-layer networks.

*   **Basic Structure:**
    An FNN consists of:
    1.  **Input Layer:** Receives the raw input features. The number of neurons in this layer corresponds to the dimensionality of the input data.
    2.  **Hidden Layer(s):** One or more layers of neurons that perform computations and transform the input data. These layers are "hidden" because their outputs are not directly observed. The depth of the network is determined by the number of hidden layers.
    3.  **Output Layer:** Produces the final output of the network (e.g., class probabilities for classification, continuous values for regression).

    Information flows in one direction – from the input layer, through the hidden layer(s), to the output layer – without any cycles or feedback loops (hence "feedforward").

    **Diagrammatic Representation:**
    ```mermaid
    flowchart LR
    subgraph "Input Layer"
        x1["x₁"]
        x2["x₂"]
        xdots["..."]
        xn["xₙ"]
        bias1["bias"]
    end

    subgraph "Hidden Layer 1"
        h11["h₁₁"]
        h12["h₁₂"]
        hdots1["..."]
        h1k["h₁ₖ"]
        bias2["bias"]
        act1["activation"]
    end

    subgraph "Hidden Layer 2"
        h21["h₂₁"]
        h22["h₂₂"]
        hdots2["..."]
        h2p["h₂ₚ"]
        bias3["bias"]
        act2["activation"]
    end

    subgraph "Output Layer"
        o1["o₁"]
        o2["o₂"]
        odots["..."]
        om["oₘ"]
        act3["activation"]
    end

    x1 -- w --> h11
    x1 -- w --> h12
    x1 -- w --> h1k
    
    x2 -- w --> h11
    x2 -- w --> h12
    x2 -- w --> h1k
    
    xdots -- w --> h11
    xdots -- w --> h12
    xdots -- w --> h1k
    
    xn -- w --> h11
    xn -- w --> h12
    xn -- w --> h1k
    
    bias1 --> act1
    
    h11 -- w --> h21
    h11 -- w --> h22
    h11 -- w --> h2p
    
    h12 -- w --> h21
    h12 -- w --> h22
    h12 -- w --> h2p
    
    hdots1 -- w --> h21
    hdots1 -- w --> h22
    hdots1 -- w --> h2p
    
    h1k -- w --> h21
    h1k -- w --> h22
    h1k -- w --> h2p
    
    bias2 --> act2
    
    h21 -- w --> o1
    h21 -- w --> o2
    h21 -- w --> om
    
    h22 -- w --> o1
    h22 -- w --> o2
    h22 -- w --> om
    
    hdots2 -- w --> o1
    hdots2 -- w --> o2
    hdots2 -- w --> om
    
    h2p -- w --> o1
    h2p -- w --> o2
    h2p -- w --> om
    
    bias3 --> act3
    ```
    -   Each connection between neurons has an associated **weight (w)**.
    -   Each neuron (typically in hidden and output layers) has an associated **bias (b)** term.
    -   Each neuron applies an **activation function** to its weighted sum of inputs plus bias.

    **Mathematical Representation (for one neuron in a layer):**
    Let $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$ be the outputs from the previous layer (or input features).
    Let $\mathbf{w} = [w_1, w_2, \dots, w_d]^T$ be the weights of the connections to the current neuron, and $b$ be its bias.
    The **pre-activation** (or logit, net input) $z$ is:
    $$ z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b $$
    The **activation** (output) $a$ of the neuron is:
    $$ a = f(z) $$
    Where $f(\cdot)$ is the activation function.

### 2. Activation Functions

Activation functions introduce non-linearity into the network. Without non-linear activation functions, a multi-layer neural network would simply be equivalent to a linear model, regardless of its depth. Non-linearity is crucial for learning complex patterns.

*   **Motivation:**
    *   **Introduce Non-linearity:** To allow the network to learn complex mappings that go beyond linear transformations.
    *   **Control Neuron Output:** To keep the output of neurons within a certain range (e.g., sigmoid, tanh) or to enable specific behaviors (e.g., sparsity with ReLU).
    *   **Impact Gradient Flow:** The choice of activation function significantly affects how gradients propagate during backpropagation, influencing the network's trainability.

*   **Common Activation Functions:**

    1.  **Sigmoid (Logistic):**
        $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
        *   **Output Range:** (0, 1)
        *   **Pros:** Interpretable as a firing rate or probability. Smooth gradient.
        *   **Cons:**
            *   **Vanishing Gradients:** For very large positive or negative $z$, the sigmoid saturates (output close to 0 or 1), and its gradient becomes very close to zero. This can slow down or stall learning in deep networks, especially in earlier layers.
            *   **Not Zero-Centered:** Output is always positive. This can make training less efficient as gradients for weights in subsequent layers will always have the same sign (assuming positive inputs).
            *   Computationally a bit expensive (exponential function).
        *   **Usage:** Historically popular, especially in output layers for binary classification (to output probabilities). Less common in hidden layers of modern deep networks due to the vanishing gradient problem.

    2.  **Hyperbolic Tangent (Tanh):**
        $$ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2 \sigma(2z) - 1 $$
        *   **Output Range:** (-1, 1)
        *   **Pros:**
            *   **Zero-Centered Output:** This can help with training dynamics compared to sigmoid.
            *   Smooth gradient.
        *   **Cons:**
            *   **Vanishing Gradients:** Still suffers from saturation and vanishing gradients for large $|z|$.
            *   Computationally a bit expensive.
        *   **Usage:** Often preferred over sigmoid in hidden layers when a bounded, zero-centered activation is desired, but still less common than ReLU variants today.

    3.  **Rectified Linear Unit (ReLU):**
        $$ \text{ReLU}(z) = \max(0, z) $$
        *   **Output Range:** $[0, \infty)$
        *   **Pros:**
            *   **Alleviates Vanishing Gradients (for $z>0$):** The gradient is 1 for $z>0$, allowing gradients to flow better.
            *   **Computational Efficiency:** Very simple to compute (a thresholding operation).
            *   **Sparsity:** Can lead to sparse activations (many neurons outputting 0), which can be efficient and sometimes beneficial for representation.
        *   **Cons:**
            *   **Not Zero-Centered Output.**
            *   **Dying ReLU Problem:** If a neuron's input $z$ is consistently negative during training, it will always output 0, and its gradient will also be 0. This neuron effectively "dies" and stops learning. A large learning rate or a large negative bias can cause this.
        *   **Usage:** The most widely used activation function in hidden layers of deep neural networks due to its simplicity and effectiveness in mitigating vanishing gradients.

    4.  **Leaky ReLU (LReLU) and Parameterized ReLU (PReLU):**
        $$ \text{Leaky ReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \le 0 \end{cases} $$
        Where $\alpha$ is a small positive constant (e.g., 0.01 for Leaky ReLU).
        For **PReLU**, $\alpha$ is a learnable parameter.
        *   **Output Range:** $(-\infty, \infty)$
        *   **Pros:**
            *   **Addresses Dying ReLU Problem:** Allows a small, non-zero gradient when the unit is not active ($z \le 0$), preventing neurons from dying.
            *   Retains benefits of ReLU (computational efficiency, good gradient flow for $z>0$).
        *   **Cons:** Performance isn't always consistently better than ReLU; results can be empirical.
        *   **Usage:** A common alternative to ReLU, especially if dying ReLUs are suspected.

    5.  **Exponential Linear Unit (ELU):**
        $$ \text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha (e^z - 1) & \text{if } z \le 0 \end{cases} $$
        Where $\alpha > 0$ is a hyperparameter.
        *   **Output Range:** $(-\alpha, \infty)$
        *   **Pros:**
            *   Can produce negative outputs, pushing mean activations closer to zero (like tanh).
            *   Avoids dying ReLU problem and has smoother transition for negative inputs than Leaky ReLU.
            *   Reported to lead to faster learning and better generalization than ReLU in some cases.
        *   **Cons:** More computationally expensive than ReLU due to the exponential function.
        *   **Usage:** Another alternative to ReLU, particularly when negative outputs are desired.

    6.  **Softmax (as discussed in Logistic Regression):**
        $$ \text{Softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} $$
        *   **Motivation:** Used primarily in the **output layer** for multi-class classification problems. It converts a vector of raw scores (logits) into a probability distribution over $K$ classes, where each output is between 0 and 1, and all outputs sum to 1.

*   **Choosing an Activation Function:**
    *   **Hidden Layers:** ReLU is often the default starting choice. If dying ReLUs are an issue, Leaky ReLU, PReLU, or ELU can be tried. Tanh or Sigmoid are generally avoided in deep hidden layers.
    *   **Output Layer:**
        *   **Binary Classification:** Sigmoid (to output a single probability).
        *   **Multi-class Classification:** Softmax (to output a probability distribution over classes).
        *   **Regression:** No activation function (linear output) or sometimes an activation that matches the expected range of the output (e.g., ReLU if output must be non-negative).

### 3. Loss Functions (Cost Functions)

The loss function quantifies how far the network's predictions are from the true target values. The goal of training is to minimize this loss.

*   **Motivation:** To provide a measurable objective that the learning algorithm (e.g., gradient descent) can optimize. A well-chosen loss function guides the network's parameters towards values that produce accurate predictions.

*   **Common Loss Functions (many overlap with those in shallow learning):**

    1.  **Mean Squared Error (MSE) / L2 Loss:**
        $$ J(\mathbf{\theta}) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(\mathbf{x}^{(i)}) - y^{(i)})^2 $$
        Or often $\frac{1}{2m}$ for convenience.
        *   **Usage:** Primarily for **regression tasks** where the output is a continuous value.
        *   **Motivation:** Penalizes larger errors more heavily due to the squaring. Assumes Gaussian noise in the target variable from an MLE perspective.

    2.  **Mean Absolute Error (MAE) / L1 Loss:**
        $$ J(\mathbf{\theta}) = \frac{1}{m} \sum_{i=1}^{m} |h_\theta(\mathbf{x}^{(i)}) - y^{(i)}| $$
        *   **Usage:** Also for **regression tasks**.
        *   **Motivation:** Less sensitive to outliers compared to MSE. Gives a more direct measure of average error magnitude.

    3.  **Cross-Entropy Loss (Log Loss):**
        *   **Binary Cross-Entropy (BCE):**
            $$ J(\mathbf{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(\mathbf{x}^{(i)})) \right] $$
            Where $h_\theta(\mathbf{x}^{(i)})$ is the predicted probability (output of sigmoid) for the positive class.
            *   **Usage:** For **binary classification tasks**.
            *   **Motivation:** Derived from MLE assuming Bernoulli distributed target variables. Measures the dissimilarity between the true probability distribution (one-hot) and the predicted probability. Penalizes confident wrong predictions heavily. (See side note in Logistic Regression section for more details on Entropy and CE).

        *   **Categorical Cross-Entropy:**
            $$ J(\mathbf{\Theta}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(h_{\mathbf{\Theta}}(\mathbf{x}^{(i)})_k) $$
            Where $y_k^{(i)}$ is 1 if example $i$ belongs to class $k$ (one-hot encoded true label), and 0 otherwise. $h_{\mathbf{\Theta}}(\mathbf{x}^{(i)})_k$ is the predicted probability (output of softmax) for class $k$.
            *   **Usage:** For **multi-class classification tasks**.
            *   **Motivation:** Generalization of BCE to multiple classes.

    4.  **Hinge Loss:**
        $$ J(\mathbf{\theta}) = \frac{1}{m} \sum_{i=1}^{m} \max(0, 1 - y^{(i)} \cdot h_\theta(\mathbf{x}^{(i)})) $$
        Where $y^{(i)} \in \{-1, 1\}$ are the true labels, and $h_\theta(\mathbf{x}^{(i)})$ is the raw output score of the model (not a probability).
        *   **Usage:** Originally for Support Vector Machines (SVMs), but can be used for training classifiers.
        *   **Motivation:** Aims to ensure a margin of separation between classes. It penalizes predictions that are on the wrong side of the margin or within the margin.

    5.  **Kullback-Leibler (KL) Divergence:**
        $$ D_{KL}(P || Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)} $$
        *   **Usage:** Often used when comparing two probability distributions, e.g., in Variational Autoencoders (VAEs) to make the learned latent distribution close to a prior (like a standard normal distribution), or in reinforcement learning.
        *   **Motivation:** Measures the information gain/loss when approximating a true distribution $P$ with a predicted distribution $Q$. Minimizing KL divergence makes $Q$ similar to $P$. (See side note in Logistic Regression section).

### 4. Learning: Backpropagation Algorithm

Training an FNN involves finding the optimal set of weights and biases ($\mathbf{\theta}$) that minimize the chosen loss function $J(\mathbf{\theta})$. The most common algorithm for this is **Backpropagation**, which is essentially an efficient way to compute the gradients of the loss function with respect to all parameters in the network, used in conjunction with an optimization algorithm like Gradient Descent (or its variants like SGD, Adam).

*   **Motivation:**
    Directly computing the gradient $\frac{\partial J}{\partial w_{jk}^l}$ (gradient of loss $J$ w.r.t. weight $w_{jk}$ connecting neuron $k$ in layer $l-1$ to neuron $j$ in layer $l$) for every weight in a deep network can be complex. Backpropagation provides a systematic and computationally efficient method based on the chain rule of calculus.

*   **Core Idea (High-Level):**
    Backpropagation consists of two main passes:
    1.  **Forward Pass:**
        *   Input data $\mathbf{x}$ is fed into the network.
        *   Activations are computed layer by layer, from the input layer through the hidden layers to the output layer, producing the network's prediction $\hat{\mathbf{y}} = h_\theta(\mathbf{x})$.
        *   The loss $J(\mathbf{\theta})$ between the prediction $\hat{\mathbf{y}}$ and the true target $\mathbf{y}$ is calculated.

    2.  **Backward Pass (Error Propagation):**
        *   **Output Layer Gradients:** The gradient of the loss function with respect to the activations (or pre-activations) of the output layer is computed first. This is often straightforward. For example, if $J = \frac{1}{2}(\hat{y} - y)^2$ and $\hat{y} = f(z_{out})$, then $\frac{\partial J}{\partial z_{out}} = (\hat{y}-y)f'(z_{out})$.
        *   **Propagate Gradients Backwards:** The algorithm then propagates these error signals (gradients) backward through the network, layer by layer.
            *   For each layer, it calculates the gradient of the loss with respect to that layer's parameters (weights and biases) and with respect to its inputs (which are the activations of the previous layer).
            *   This is done using the **chain rule**. For example, to find the gradient w.r.t. a weight $w$ in an earlier layer, we consider how $w$ affects its neuron's pre-activation $z$, how $z$ affects its activation $a$, how $a$ (as an input to the next layer) affects the loss indirectly through all subsequent layers.
            $$ \frac{\partial J}{\partial w_{jk}^l} = \frac{\partial J}{\partial a_j^l} \frac{\partial a_j^l}{\partial z_j^l} \frac{\partial z_j^l}{\partial w_{jk}^l} $$
            The term $\frac{\partial J}{\partial a_j^l}$ (or $\frac{\partial J}{\partial z_j^l}$, often denoted as $\delta_j^l$ or "error term") for layer $l$ is computed using the "error terms" from layer $l+1$.
        *   **Parameter Updates:** Once all gradients $\nabla J(\mathbf{\theta})$ are computed, an optimization algorithm (like SGD) updates the parameters:
            $$ \mathbf{\theta} := \mathbf{\theta} - \alpha \nabla J(\mathbf{\theta}) $$

*   **Key Steps in Derivation (Conceptual):**
    1.  Define $\delta_j^L = \frac{\partial J}{\partial z_j^L}$ for an output neuron $j$ in the output layer $L$. This can be calculated directly.
    2.  Relate $\delta_j^l$ (error at neuron $j$ in layer $l$) to the errors in the next layer $l+1$:
        $$ \delta_j^l = \left( \sum_k w_{kj}^{l+1} \delta_k^{l+1} \right) f'(z_j^l) $$
        This equation shows how errors are "backpropagated" weighted by the connections $w_{kj}^{l+1}$. $f'(z_j^l)$ is the derivative of the activation function at neuron $j$ in layer $l$.
    3.  Calculate gradients for weights and biases using these $\delta$ terms:
        $$ \frac{\partial J}{\partial w_{jk}^l} = a_k^{l-1} \delta_j^l $$
        $$ \frac{\partial J}{\partial b_j^l} = \delta_j^l $$
        Where $a_k^{l-1}$ is the activation of neuron $k$ in the previous layer $l-1$.

*   **Efficiency:** Backpropagation avoids redundant calculations by dynamically computing and reusing intermediate gradient values. It's much more efficient than naively applying the chain rule to each parameter independently.

*   **Implementation:** Deep learning frameworks like PyTorch and TensorFlow have automatic differentiation engines (e.g., Autograd in PyTorch) that automatically compute these gradients during the `loss.backward()` call, so developers rarely need to implement backpropagation manually. However, understanding its principles is crucial for debugging and designing effective networks.

---

## Appendix: FNN Implementation with PyTorch

This appendix demonstrates Feedforward Neural Network (FNN or MLP) implementations using PyTorch for both classification and regression tasks.

### 1. FNN for Image Classification (MNIST Dataset)

This example demonstrates a complete Feedforward Neural Network (FNN or MLP) implementation using PyTorch for classifying handwritten digits from the MNIST dataset.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 0. Device Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# 1. Hyperparameters
input_size = 28 * 28  # MNIST images are 28x28 pixels
hidden_size1 = 512
hidden_size2 = 256
num_classes = 10      # Digits 0-9
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# 2. MNIST Dataset Loading and Preprocessing
# Transformation: Convert images to PyTorch tensors and normalize pixel values
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std Dev for MNIST
])

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Let's look at one batch of images
examples = iter(test_loader)
example_data, example_targets = next(examples)

# For plotting an image from the batch
# def imshow(img):
#     img = img * 0.3081 + 0.1307  # Unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze(), cmap='gray')
#     plt.show()
# imshow(torchvision.utils.make_grid(example_data[:4]))
# print('GroundTruth: ', ' '.join(f'{example_targets[j]}' for j in range(4)))


# 3. FNN Model Definition
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(FeedForwardNN, self).__init__()
        # Layer 1: Input to Hidden1
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU() # Activation for hidden layer 1
        
        # Layer 2: Hidden1 to Hidden2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU() # Activation for hidden layer 2
        
        # Layer 3: Hidden2 to Output
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        # No activation here, as nn.CrossEntropyLoss will apply log_softmax internally

    def forward(self, x):
        # Flatten the image: (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.reshape(-1, 28*28) 
        
        # Pass through first hidden layer
        out = self.fc1(x)
        out = self.relu1(out)
        
        # Pass through second hidden layer
        out = self.fc2(out)
        out = self.relu2(out)
        
        # Pass through output layer
        out = self.fc3(out) # Output raw scores (logits)
        return out

model = FeedForwardNN(input_size, hidden_size1, hidden_size2, num_classes).to(device)

# 4. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss() # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam is a popular optimizer

# 5. Training Loop
print("\n--- Starting Training ---")
n_total_steps = len(train_loader)
training_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images) # outputs are logits
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights

        if (i + 1) % (n_total_steps // 5) == 0: # Print roughly 5 times per epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    
    avg_epoch_loss = epoch_loss / n_total_steps
    training_losses.append(avg_epoch_loss)
    print(f'--- Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f} ---')

print("--- Training Finished ---")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.show()

# 6. Testing and Evaluation
# In test phase, we don't need to compute gradients (for memory efficiency)
print("\n--- Starting Testing ---")
with torch.no_grad(): # Disables gradient calculation
    n_correct = 0
    n_samples = 0
    all_labels = []
    all_predictions = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images) # Logits
        
        # Max returns (value, index)
        _, predicted_classes = torch.max(outputs.data, 1) # Get the index of the max logit
        
        n_samples += labels.size(0)
        n_correct += (predicted_classes == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted_classes.cpu().numpy())

    accuracy = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

# Optional: More detailed evaluation (e.g., confusion matrix)
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, digits=4))

conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 7. Example Predictions (Optional)
print("\n--- Example Predictions ---")
dataiter = iter(test_loader)
images_test, labels_test = next(dataiter)
images_test_on_device = images_test.to(device)

with torch.no_grad():
    outputs_test = model(images_test_on_device)
    _, predicted_test = torch.max(outputs_test, 1)

# Function to show an image
def imshow_mnist(img_tensor, title=None):
    img_tensor = img_tensor / 2 + 0.5 # Unnormalize if normalization was (0.5, 0.5)
                                      # For MNIST (0.1307, 0.3081), simple unnorm is fine for viz
    img_numpy = img_tensor.cpu().numpy()
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)).squeeze(), cmap='gray')
    if title:
        plt.title(title)
    # plt.show() # Use plt.show() after all subplots are drawn

plt.figure(figsize=(12, 5))
for i in range(8): # Show first 8 images from the batch
    plt.subplot(2, 4, i + 1)
    imshow_mnist(images_test[i], title=f"Pred: {predicted_test[i].item()}, True: {labels_test[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

---

### 2. FNN for Regression (California Housing Dataset)

This example demonstrates using an FNN for a regression task: predicting median house values in California districts.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# 0. Device Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# 1. Hyperparameters for Regression
# These might need tuning based on the dataset and model complexity
# input_size_reg will be determined by the dataset
hidden_size_reg1 = 128
hidden_size_reg2 = 64
output_size_reg = 1   # Predicting a single continuous value (median house value)
num_epochs_reg = 100
batch_size_reg = 64
learning_rate_reg = 0.001

# 2. California Housing Dataset Loading and Preprocessing
# Fetch dataset
housing = fetch_california_housing()
X_numpy_reg, y_numpy_reg = housing.data, housing.target

# Reshape y to be a column vector
y_numpy_reg = y_numpy_reg.reshape(-1, 1)

# Split data
X_train_np_reg, X_test_np_reg, y_train_np_reg, y_test_np_reg = train_test_split(
    X_numpy_reg, y_numpy_reg, test_size=0.2, random_state=42
)

# Scale features (important for neural networks)
scaler_reg = StandardScaler()
X_train_np_reg = scaler_reg.fit_transform(X_train_np_reg)
X_test_np_reg = scaler_reg.transform(X_test_np_reg)

# Convert to PyTorch tensors
X_train_reg = torch.from_numpy(X_train_np_reg.astype(np.float32)).to(device)
y_train_reg = torch.from_numpy(y_train_np_reg.astype(np.float32)).to(device)
X_test_reg = torch.from_numpy(X_test_np_reg.astype(np.float32)).to(device)
y_test_reg = torch.from_numpy(y_test_np_reg.astype(np.float32)).to(device)

# Create TensorDatasets and DataLoaders
train_dataset_reg = TensorDataset(X_train_reg, y_train_reg)
test_dataset_reg = TensorDataset(X_test_reg, y_test_reg)

train_loader_reg = DataLoader(dataset=train_dataset_reg, batch_size=batch_size_reg, shuffle=True)
test_loader_reg = DataLoader(dataset=test_dataset_reg, batch_size=batch_size_reg, shuffle=False)

input_size_reg = X_train_reg.shape[1] # Number of features

# 3. FNN Model Definition for Regression
class RegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim) # Output layer for regression is typically linear

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out) # No activation for regression output
        return out

model_reg = RegressionNN(input_size_reg, hidden_size_reg1, hidden_size_reg2, output_size_reg).to(device)

# 4. Loss Function and Optimizer for Regression
criterion_reg = nn.MSELoss() # Mean Squared Error for regression
optimizer_reg = optim.Adam(model_reg.parameters(), lr=learning_rate_reg)

# 5. Training Loop for Regression
print("\n--- Starting Regression Training ---")
n_total_steps_reg = len(train_loader_reg)
training_losses_reg = []
for epoch in range(num_epochs_reg):
    model_reg.train() # Set model to training mode
    epoch_loss_reg = 0
    for i, (features, targets) in enumerate(train_loader_reg):
        features = features.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs_reg = model_reg(features)
        loss_reg = criterion_reg(outputs_reg, targets)
        epoch_loss_reg += loss_reg.item()

        # Backward pass and optimization
        optimizer_reg.zero_grad()
        loss_reg.backward()
        optimizer_reg.step()

        # if (i + 1) % (n_total_steps_reg // 2) == 0 : # Print a few times per epoch
        #     print(f'Reg - Epoch [{epoch+1}/{num_epochs_reg}], Step [{i+1}/{n_total_steps_reg}], Loss: {loss_reg.item():.4f}')
    
    avg_epoch_loss_reg = epoch_loss_reg / n_total_steps_reg
    training_losses_reg.append(avg_epoch_loss_reg)
    if (epoch + 1) % 10 == 0 or epoch == 0:
      print(f'--- Reg - Epoch [{epoch+1}/{num_epochs_reg}] Average Training Loss: {avg_epoch_loss_reg:.4f} ---')

print("--- Regression Training Finished ---")

# Plot training loss for regression
plt.figure(figsize=(10, 5))
plt.plot(training_losses_reg, label='Regression Training Loss (MSE)')
plt.title('Regression Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.legend()
plt.grid(True)
plt.show()

# 6. Testing and Evaluation for Regression
print("\n--- Starting Regression Testing ---")
model_reg.eval() # Set model to evaluation mode
all_predictions_reg = []
all_targets_reg = []
test_loss_mse = 0
num_test_batches = 0

with torch.no_grad():
    for features, targets in test_loader_reg:
        features = features.to(device)
        targets = targets.to(device)
        
        outputs_reg = model_reg(features)
        loss_reg = criterion_reg(outputs_reg, targets)
        test_loss_mse += loss_reg.item()
        num_test_batches +=1

        all_predictions_reg.extend(outputs_reg.cpu().numpy())
        all_targets_reg.extend(targets.cpu().numpy())

avg_test_loss_mse = test_loss_mse / num_test_batches
print(f'Average Test MSE: {avg_test_loss_mse:.4f}')

# Convert lists to numpy arrays for easier plotting
all_predictions_reg = np.array(all_predictions_reg).flatten()
all_targets_reg = np.array(all_targets_reg).flatten()

# Calculate R-squared (coefficient of determination)
# Note: This is a common regression metric, not directly optimized but good for evaluation
from sklearn.metrics import r2_score
r2 = r2_score(all_targets_reg, all_predictions_reg)
print(f'Test R-squared: {r2:.4f}')

# 7. Visualization of Regression Predictions vs. Actual
plt.figure(figsize=(10, 6))
plt.scatter(all_targets_reg, all_predictions_reg, alpha=0.5, s=10, label='Predictions vs Actual')
# Plot a line for perfect predictions (y=x)
min_val = min(all_targets_reg.min(), all_predictions_reg.min())
max_val = max(all_targets_reg.max(), all_predictions_reg.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction Line')
plt.xlabel("Actual Median House Value ($100,000s)")
plt.ylabel("Predicted Median House Value ($100,000s)")
plt.title("FNN Regression: Predictions vs. Actual Values (California Housing)")
plt.legend()
plt.grid(True)
plt.show()

# Plot residuals (optional, good for diagnosing model fit)
residuals = all_targets_reg - all_predictions_reg
plt.figure(figsize=(10, 6))
plt.scatter(all_predictions_reg, residuals, alpha=0.5, s=10)
plt.hlines(0, xmin=all_predictions_reg.min(), xmax=all_predictions_reg.max(), colors='r', linestyles='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.grid(True)
plt.show()
```