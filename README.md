# Welcome to the Machine

![wttm](assets/media/cover.jpeg)

---

## Table of Contents

- [**Tips**](notes/tips.md)
  - Useful resources for learning Python and AI
  - Some tips for python coding
  - Some recommended tools

- **Prequisite(Math)**
  - Linear Algebra
  - Matrix Calculus
  - Probability Theory
  - Mathematical Statistics

- **Machine Learning/Statistical Learning**
  - [ ] [**Overview**](notes/machine-learning_statistical-learning/overview.md)
    - Categories of ML models
      - Supervised vs. Unsupervised
      - Generative vs. Discriminative
      - Parametric vs. Non-parametric
    - Representation, Inference, and Learning
  - [ ] [**Regression**](notes/machine-learning_statistical-learning/regression.md)
    - Polynomial Regression
    - Linear Regression
      - Why MSE(Mean Squared Error)?
      - Bias-Variance Tradeoff
        - Overfitting and Underfitting
      - Regularization
        - Ridge, Lasso
        - MLE(Maximum Likelihood Estimation) and MAP(Maximum A Posteriori)
      - Optimization
        - Gradient Descent
        - Stochastic Gradient Descent
    - Logistic Regression
      - logit, logistic, regression
      - Sigmoid(Logistic) and Softmax
      - Why CE(Cross Entropy)?
      - Criterion on classification
    - MLE, CE, and KL Divergence
  - **$ \cdots $ 🏗️ $ \text{work in progress} $ 🏗️ $ \cdots $**
- **Deep Learning**
  - [ ] **Overview**
    - Why Deep Learning?
      - Deep Learning vs. Shallow Learning
      - Challenges Motivating Deep Learning
    - NN as Universal Approximation
  - [ ] **FNN(Feedforward Neural Network)**
    - Origin
    - Activation functions
      - Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish, GELU, Softmax
    - Loss functions
      - MSE, CE, KL Divergence
    - Learning
      - Backpropagation
  - [ ] **CNN(Convolutional Neural Network)**
    - Convolution?
      - continuous and discrete convolution
    - Get your hands dirty(some calculation in CNNs)
      - Convolution layer
        - 1D, 2D, 3D
        - stride, padding, dilation
        - kernel size
      - Pooling layer
        - Max Pooling, Average Pooling
      - Batch Normalization
    - Examples
      - LeNet, AlexNet, VGG, ResNet
  - [ ] **RNN(Recurrent Neural Network)\***
    - Sequence modeling
    - LSTM
    - GRU
  - **$ \cdots $ 🏗️ $ \text{work in progress} $ 🏗️ $ \cdots $**


- **Implementation**
  - [ ] [`numpy`](https://numpy.org) scientific computing
    - N-dimensional array
    - Broadcasting
    - Vectorization
    - Linear Algebra
      - Matrix Multiplication
      - Singular Value Decomposition(SVD)
      - Eigenvalue and Eigenvector

  - [ ] [`pytorch`](https://pytorch.org) machine learning framework
    - Tensors
    - [Computational Graphs and Autograd](https://github.com/Paperspace/PyTorch-101-Tutorial-Series/blob/master/PyTorch%20101%20Part%201%20-%20Computational%20Graphs%20and%20Autograd%20in%20PyTorch.ipynb)
    - Optimizers
    - DataLoader
    - Neural Networks